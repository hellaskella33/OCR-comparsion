import json
import pickle
import logging
import time
import traceback
from random import random
from typing import List
from multiprocessing import Pool

import tqdm
import os
import argparse
import requests
import pandas as pd

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from functools import partial

from paddleocr import PaddleOCR

from ai.dates_extraction import extract_dates, filter_dates_out_of_range, format_dates, filter_invalid_dates
from models.document import Bookmark, Document
from settings import BE_RESULTS_ENDPOINT, DATA_PATH, LOCAL_IMAGES_PATH, CONFIG, \
    DATABASE_URI, BE_DELETE_IMAGES_ENDPOINT, PROCESSES_COUNT, LOGS_PATH, CREATED_BY_AI_TAG
from train import load_last_trained_model
from utils.logger import initiate_module_logger
from utils.utils import notify_when_ends

initiate_module_logger(LOGS_PATH, __file__, console_level="info")

# Initialize PaddleOCR
ocr = PaddleOCR(
    rec_model_dir='/home/maxkhamuliak/projects/OCR-comparsion/recognition_folder/en_PP-OCRv4_rec_train',
    det_model_dir='/home/maxkhamuliak/projects/OCR-comparsion/detection_models/en_PP-OCRv3_det_slim_distill_train',
    use_angle_cls=True,
    lang='en'
)

def send_bookmarks_to_be(document: Document):
    logging.info(f"POST:\n{BE_RESULTS_ENDPOINT}\n\nJSON:\n{document.to_json()}")
    try:
        r = requests.post(BE_RESULTS_ENDPOINT, verify=False, json=document.to_json())
        logging.info(f"BE response: {r.text}")
    except Exception as e:
        logging.info(f"Failed POST:\n{traceback.format_exc()}")

def send_delete_images_to_be(document: Document):
    logging.info(f"Sent delete images request for {document.document_id}")
    r = requests.delete(BE_DELETE_IMAGES_ENDPOINT, verify=False, json={"documentId": document.document_id})
    logging.info(f"BE status: {r.status_code} response: {r.text}")

def predict_bookmarks(df: pd.DataFrame, random=False):
    trained_model = load_last_trained_model()
    if random:
        return trained_model.random_predict(df)
    return trained_model.predict(df)

def extract_bookmarks_from_candidates_df(df: pd.DataFrame) -> dict:
    bookmarks = {}
    for i, r in df.iterrows():
        for bookmark in r["bookmarks&confidence"]:
            bookmark_o = Bookmark(levels=[], dates=r["dates"], confidence_score=max(bookmark[1], 0.01))
            for name in bookmark[0].split("@@@"):
                bookmark_o.add_sublevel(name)
            bookmarks[i] = bookmarks.get(i, []) + [bookmark_o]
        if len(r['bookmarks&confidence']) == 0:
            bookmark_o = Bookmark(levels=[], dates=r["dates"], confidence_score=0)
            bookmarks[i] = bookmarks.get(i, []) + [bookmark_o]
    return bookmarks

def process_image(filename):
    filepath = os.path.join(LOCAL_IMAGES_PATH, filename)
    result = ocr.ocr(filepath, cls=True)
    text = ' '.join([line[1][0] for line in result[0]])
    logging.info(f"{filename} processed")
    return text

@notify_when_ends
def cast_images_to_text(images_path) -> pd.DataFrame:
    p = Pool(processes=PROCESSES_COUNT)
    resulted_text = {"filename": [], "page_index": [], "text": []}
    
    for i, flnm in tqdm.tqdm(enumerate(sorted(os.listdir(images_path),
                                              key=lambda x: list(map(int, x.split(".")[0].split("_")[2:])))),
                             desc=f"Processing images from: {images_path}"):
        resulted_text["filename"].append(flnm)
        resulted_text["page_index"].append(i)

    resulted_text["text"] = list(tqdm.tqdm(p.imap(process_image, resulted_text["filename"])))
    return pd.DataFrame.from_dict(resulted_text)

def main(args):
    if args.send_bookmarks_to_be:
        logging.info("I will send bookmarks")

    logging.info(f"Started processing {args.document_id}")
    logging.info(f"Started extracting text from {args.images_path} images")
    document_df = cast_images_to_text(os.path.join(LOCAL_IMAGES_PATH, args.images_path))

    document_df["provider"] = args.provider
    document_df["document_id"] = args.document_id
    document_df["dates"] = document_df.text.map(extract_dates).map(
        filter_dates_out_of_range(CONFIG.DATE_RANGE_FROM_TODAY_DATE)).map(filter_invalid_dates).map(format_dates)
    document_df["bookmarks&confidence"] = predict_bookmarks(document_df)
    document = Document(args.document_id, args.cclr_id, args.provider, document_df["text"],
                        extract_bookmarks_from_candidates_df(document_df))

    engine = create_engine(DATABASE_URI, echo=False, future=True)
    Session = sessionmaker(bind=engine)
    document.doc_save_to_db(Session, created_by=CREATED_BY_AI_TAG, store_only_document=False)

    if args.send_bookmarks_to_be:
        send_bookmarks_to_be(document)
    else:
        send_delete_images_to_be(document)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", help="path to the images of document", type=str, default="test")
    parser.add_argument("--provider", type=str, default="unknown")
    parser.add_argument("--cclr_id", type=str, default=0)
    parser.add_argument("--document_id", type=str, default="".join([str(random()) for i in range(10)]))
    parser.add_argument("--send_bookmarks_to_be", action='store_true', default=False)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logging.error(e, exc_info=True)
        logging.info("retrying")
        time.sleep(60)
        main(args)