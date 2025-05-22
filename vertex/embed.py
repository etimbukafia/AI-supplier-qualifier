import os
import logging
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase
from pinecone import Pinecone
from dotenv import load_dotenv
from google.genai import Client
from datetime import datetime
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Connection settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("ingest.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Batch size
BATCH_SIZE = 100

# Initialize clients
gemini_client = Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX)


def init_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.supplier_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:AuditReport) REQUIRE a.audit_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (q:QualityLog) REQUIRE q.log_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Delivery) REQUIRE d.delivery_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RiskIndicator) REQUIRE r.risk_id IS UNIQUE"
    ]
    with driver.session() as session:
        for c in constraints:
            try:
                session.run(c)
                logger.info(f"Constraint created or exists: {c}")
            except Exception as e:
                logger.error(f"Error creating constraint {c}: {e}")
    return driver


def embed_text(text: str):
    try:
        resp = gemini_client.models.embed_content(model="text-embedding-004", contents=text)
        return resp.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding failure: {e}")
        return None


def parse_date(s):
    if pd.isna(s) or not str(s).strip():
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    logger.warning(f"Unparsed date: {s}")
    return None


def batcher(df):
    total = len(df)
    for i in range(0, total, BATCH_SIZE):
        yield df.iloc[i:i + BATCH_SIZE]


def ingest_audit(driver):
    df = pd.read_csv(DATA_DIR/"audit_reports.csv").fillna("")
    logger.info(f"Audit reports: {len(df)} records")
    batches = list(batcher(df))
    with driver.session() as session:
        for batch in tqdm(batches, desc="Audit batches", total=len(batches)):
            pinecone_batch = []
            tx = session.begin_transaction()
            for _, r in batch.iterrows():
                ad = parse_date(r.audit_date)
                dd = parse_date(r.due_date)
                cd = parse_date(r.closure_date)
                tx.run("MERGE (s:Supplier {supplier_id:$sid})", sid=r.supplier_id)
                tx.run(
                    "MATCH (s:Supplier {supplier_id:$sid}) CREATE (a:AuditReport {audit_id:$aid, audit_date:date($ad), standard:$std, clause:$cl, finding_type:$ft, severity:$sv, description:$desc, corrective_action:$ca, due_date:date($dd), closure_date:date($cd), status:$st})-[:FOR_SUPPLIER]->(s)",
                    sid=r.supplier_id, aid=r.audit_id, ad=ad, std=r.standard, cl=r.clause, ft=r.finding_type, sv=r.severity, desc=r.description, ca=r.corrective_action_description, dd=dd, cd=cd, st=r.status
                )
                emb = embed_text(f"{r.audit_id}: {r.description} {r.corrective_action_description}")
                if emb:
                    pinecone_batch.append({'id':f"audit_{r.audit_id}",'values':emb,'metadata':{'audit_id':r.audit_id}})
            tx.commit()
            if pinecone_batch:
                try:
                    pinecone_index.upsert(vectors=pinecone_batch)
                except Exception as e:
                    logger.error(f"Pinecone upsert error: {e}")


def ingest_quality(driver):
    df = pd.read_csv(DATA_DIR/"quality_logs.csv").fillna(0)
    logger.info(f"Quality logs: {len(df)} records")
    batches = list(batcher(df))
    with driver.session() as session:
        for batch in tqdm(batches, desc="Quality batches", total=len(batches)):
            tx = session.begin_transaction()
            for _, r in batch.iterrows():
                lid = f"QL_{r.supplier_id}_{r.year_month}"
                tx.run("MERGE (s:Supplier {supplier_id:$sid})", sid=r.supplier_id)
                tx.run(
                    "MATCH (s:Supplier {supplier_id:$sid}) CREATE (q:QualityLog {log_id:$lid, year_month:$ym, defect_ppm:$dppm, rework_pct:$rwp, ncr_count:$nc})-[:LOG_OF]->(s)",
                    sid=r.supplier_id, lid=lid, ym=r.year_month, dppm=r.defect_rate_ppm, rwp=r.rework_percentage, nc=r.ncr_count
                )
            tx.commit()


def ingest_delivery(driver):
    df = pd.read_csv(DATA_DIR/"delivery_performance.csv").fillna(0)
    logger.info(f"Delivery records: {len(df)}")
    batches = list(batcher(df))
    with driver.session() as session:
        for batch in tqdm(batches, desc="Delivery batches", total=len(batches)):
            tx = session.begin_transaction()
            for _, r in batch.iterrows():
                did = f"DL_{r.supplier_id}_{r.year_month}"
                tx.run("MERGE (s:Supplier {supplier_id:$sid})", sid=r.supplier_id)
                tx.run(
                    "MATCH (s:Supplier {supplier_id:$sid}) CREATE (d:Delivery {delivery_id:$did, year_month:$ym, on_time:$otd, promised_lt:$plt, actual_lt:$alt, deviation:$dev})-[:DELIVERED_BY]->(s)",
                    sid=r.supplier_id, did=did, ym=r.year_month, otd=r.on_time_delivery_pct, plt=r.promised_lead_time_days, alt=r.actual_lead_time_days, dev=r.schedule_deviation_days
                )
            tx.commit()


def ingest_risk(driver):
    df = pd.read_csv(DATA_DIR/"public_risk_indicators.csv").fillna("")
    logger.info(f"Risk indicators: {len(df)} records")
    batches = list(batcher(df))
    with driver.session() as session:
        for batch in tqdm(batches, desc="Risk batches", total=len(batches)):
            tx = session.begin_transaction()
            for _, r in batch.iterrows():
                rid = f"RI_{r.supplier_id}_{r.assessment_date}"
                ad = parse_date(r.assessment_date)
                tx.run("MERGE (s:Supplier {supplier_id:$sid})", sid=r.supplier_id)
                # Use MERGE on RiskIndicator to avoid duplicates
                tx.run(
                    "MATCH (s:Supplier {supplier_id:$sid}) " 
                    "MERGE (ri:RiskIndicator {risk_id:$rid}) " 
                    "SET ri.country=$ct, ri.assessment_date=date($ad), ri.current_ratio=$cr, " 
                    "ri.debt_to_equity_ratio=$de, ri.altman_z_score=$az, ri.credit_rating=$crat, " 
                    "ri.geopolitical_risk_score=$grs, ri.sanctions_flag=$sf, ri.trade_embargo_flag=$tef, " 
                    "ri.esg_controversy_score=$ecs, ri.anti_bribery_compliance=$abc, ri.trade_compliance=$tc, ri.sanctions_compliance=$sc " 
                    "MERGE (ri)-[:RISK_OF]->(s)",
                    sid=r.supplier_id, rid=rid, ct=r.country, ad=ad, cr=r.current_ratio, de=r.debt_to_equity_ratio,
                    az=r.altman_z_score, crat=r.credit_rating, grs=r.geopolitical_risk_score, sf=r.sanctions_flag,
                    tef=r.trade_embargo_flag, ecs=r.esg_controversy_score, abc=r.anti_bribery_compliance,
                    tc=r.trade_compliance, sc=r.sanctions_compliance
                )
            tx.commit()


if __name__ == "__main__":
    driver = init_neo4j()
    #ingest_audit(driver)
    #ingest_quality(driver)
    #ingest_delivery(driver)
    ingest_risk(driver)
    driver.close()
    logger.info("All data ingested successfully.")







# === Cypher Notes ===
# Constraints & Indexes (run once):
# CREATE CONSTRAINT ON (s:Supplier) ASSERT s.supplier_id IS UNIQUE;
# CREATE INDEX ON :AuditReport(audit_date);
# CREATE INDEX ON :QualityLog(year_month);
# CREATE INDEX ON :Delivery(year_month);
# CREATE INDEX ON :RiskIndicator(country);

# Full-text index example:
# CALL db.index.fulltext.createNodeIndex("auditDescriptions", ["AuditReport"], ["description"]);

# Vector index in Neo4j (Enterprise/GDS):
# CALL gds.beta.vector.index.create("auditEmbeddings", { nodeProjection: "AuditReport", embeddingProperty: "embedding", similarityMetric: "COSINE" });

