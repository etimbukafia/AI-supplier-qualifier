import os
import logging
from neo4j import GraphDatabase
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─── Clients ──────────────────────────────────────────────────────────────────
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
pc = Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index(PINECONE_INDEX)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── Helper Functions ─────────────────────────────────────────────────────────

def fetch_structured(supplier_id: str):
    """Aggregate all metrics from Neo4j for the last 12 months without APOC."""
    with neo4j_driver.session() as sess:
        rec = sess.run("""
        MATCH (s:Supplier {supplier_id: $sid})
        // 1) Audit findings
        OPTIONAL MATCH (s)<-[:FOR_SUPPLIER]-(a:AuditReport)
        WITH s,
             SUM(CASE WHEN a.severity = 'Major' THEN 1 ELSE 0 END) AS open_majors,
             SUM(CASE WHEN a.severity = 'Minor' THEN 1 ELSE 0 END) AS open_minors
        // 2) Quality logs
        OPTIONAL MATCH (s)<-[:LOG_OF]-(q:QualityLog)
        WHERE q.year_month >= $year_month_threshold
        WITH s, open_majors, open_minors,
             AVG(q.defect_ppm) AS avg_ppm,
             AVG(q.rework_pct) AS avg_rework,
             SUM(q.ncr_count) AS total_ncr
        // 3) Delivery performance
        OPTIONAL MATCH (s)<-[:DELIVERED_BY]-(d:Delivery)
        WHERE d.year_month >= $year_month_threshold
        WITH s, open_majors, open_minors, avg_ppm, avg_rework, total_ncr,
             AVG(d.on_time) AS avg_otd,
             AVG(d.deviation) AS avg_dev
        // 4) Risk indicators
        OPTIONAL MATCH (s)<-[:RISK_OF]-(r:RiskIndicator)
        RETURN open_majors, open_minors,
               avg_ppm, avg_rework, total_ncr,
               avg_otd, avg_dev,
               r.altman_z_score AS fin_score,
               r.geopolitical_risk_score AS geo_score,
               r.sanctions_flag AS sanctioned,
               r.trade_embargo_flag AS embargoed,
               r.esg_controversy_score AS esg_score,
               r.anti_bribery_compliance AS acb,
               r.trade_compliance AS tcomp,
               r.sanctions_compliance AS scomp
        """, sid=supplier_id, year_month_threshold=calculate_year_month_threshold())
        
        # Get the first record and handle None values
        records = rec.data()
        result = records[0] if records else None
        
        if result:
            return {
                'open_majors': result['open_majors'] or 0,
                'open_minors': result['open_minors'] or 0,
                'avg_ppm': result['avg_ppm'] or 0.0,
                'avg_rework': result['avg_rework'] or 0.0,
                'total_ncr': result['total_ncr'] or 0,
                'avg_otd': result['avg_otd'] or 0.0,
                'avg_dev': result['avg_dev'] or 0.0,
                'fin_score': result['fin_score'] or 0,
                'geo_score': result['geo_score'] or 0,
                'sanctioned': result['sanctioned'] or False,
                'embargoed': result['embargoed'] or False,
                'esg_score': result['esg_score'] or 0,
                'acb': result['acb'] or 'Unknown',
                'tcomp': result['tcomp'] or 'Unknown',
                'scomp': result['scomp'] or 'Unknown'
            }
        else:
            # Return default values if no supplier found
            return {
                'open_majors': 0,
                'open_minors': 0,
                'avg_ppm': 0.0,
                'avg_rework': 0.0,
                'total_ncr': 0,
                'avg_otd': 0.0,
                'avg_dev': 0.0,
                'fin_score': 0,
                'geo_score': 0,
                'sanctioned': False,
                'embargoed': False,
                'esg_score': 0,
                'acb': 'Unknown',
                'tcomp': 'Unknown',
                'scomp': 'Unknown'
            }

def calculate_year_month_threshold():
    """Calculate the year_month threshold string (e.g., '202404') for last 12 months filtering."""
    from datetime import datetime, timedelta

    today = datetime.today()
    year = today.year
    month = today.month

    # Calculate year-month 12 months ago
    month -= 12
    if month <= 0:
        month += 12
        year -= 1

    # Format as 'YYYYMM' (assuming year_month field is stored like this)
    return f"{year}{month:02d}"

def fetch_narratives(supplier_id: str, top_k: int = 5):
    """Pull top-K most relevant audit/comment snippets from Pinecone."""
    try:
        resp = pc_index.query(
            vector=[0]*768,
            filter={"metadata.supplier_id": supplier_id},
            top_k=top_k,
            include_metadata=True
        )
        return [m['metadata']['snippet'] for m in resp['matches'] if 'metadata' in m and 'snippet' in m['metadata']]
    except Exception as e:
        logging.error(f"Error fetching narratives: {e}")
        return []

def build_prompt(metrics: dict, snippets: list, supplier_id: str, need: str) -> str:
    """Construct a detailed prompt leveraging ALL data."""
    prompt = [
        "System: You are an expert supply-chain risk analyst.",
        f"Your task: Evaluate whether to APPROVE or REJECT the supplier {supplier_id} for supplying {need},",
        "based on quantitative KPIs, compliance flags, and auditor comments.",
        "",
        f"Supplier: {supplier_id}",
        "",
        "1) Audit Findings (all-time):",
        f"   • Open Major findings: {metrics['open_majors']}",
        f"   • Open Minor findings: {metrics['open_minors']}",
        "",
        "2) Quality (last 12 mo):",
        f"   • Avg Defect Rate: {metrics['avg_ppm']:.1f} PPM",
        f"   • Avg Rework: {metrics['avg_rework']:.1f} %",
        f"   • Total NCRs: {metrics['total_ncr']}",
        "",
        "3) Delivery (last 12 mo):",
        f"   • On-time delivery: {metrics['avg_otd']:.1f} %",
        f"   • Avg schedule deviation: {metrics['avg_dev']:.1f} days",
        "",
        "4) External Risk & Compliance:",
        f"   • Financial health score: {metrics['fin_score']}/100",
        f"   • Geopolitical risk score: {metrics['geo_score']}/100",
        f"   • Sanctioned: {metrics['sanctioned']}",
        f"   • Trade embargoed: {metrics['embargoed']}",
        f"   • ESG controversy score: {metrics['esg_score']}/100",
        f"   • Anti-bribery compliant: {metrics['acb']}",
        f"   • Trade compliant: {metrics['tcomp']}",
        f"   • Sanctions compliant: {metrics['scomp']}",
        "",
        "5) Top Auditor Comments:",
    ]
    
    if snippets:
        prompt += [f"   {i+1}. {sn}" for i, sn in enumerate(snippets)]
    else:
        prompt.append("   No auditor comments available.")
    
    prompt += [
        "",
        "Guidelines:",
        "- REJECT if any of these is true:",
        "  • open_majors ≥ 1",
        "  • avg_ppm > 1000 PPM",
        "  • total_ncr > 2",
        "  • sanctioned = True OR embargoed = True",
        "- Otherwise, APPROVE.",
        "",
        "Respond with a clear decision (APPROVE/REJECT) followed by a well-constructed paragraph explaining your reasoning based on the data provided."
    ]
    return "\n".join(prompt)

def make_final_decision(metrics: dict, gemini_decision: str, supplier_id: str) -> dict:
    """
    Multi-tier decision making with rule hierarchy, risk scoring, and AI synthesis.
    Returns comprehensive decision with reasoning and confidence levels.
    """
    decision_result = {
        'final_decision': None,
        'decision_tier': None,
        'confidence_level': None,
        'reasoning': [],
        'risk_factors': [],
        'mitigation_suggestions': [],
        'review_required': False
    }
    
    # ═══════════════════════════════════════════════════════════════
    # TIER 1: ABSOLUTE DISQUALIFIERS (No exceptions)
    # ═══════════════════════════════════════════════════════════════
    tier1_violations = []
    
    if metrics['sanctioned']:
        tier1_violations.append("Supplier is on sanctions list")
    if metrics['embargoed']:
        tier1_violations.append("Supplier subject to trade embargo")
    if metrics['scomp'] in ['Non-Compliant', 'Under Investigation']:  # Changed from boolean check
        tier1_violations.append("Sanctions compliance failure")
    if metrics['open_majors'] >= 5:
        tier1_violations.append(f"Critical audit failures ({metrics['open_majors']} major findings)")
    
    if tier1_violations:
        decision_result.update({
            'final_decision': 'REJECT',
            'decision_tier': 'TIER_1_ABSOLUTE',
            'confidence_level': 'CERTAIN',
            'reasoning': tier1_violations,
            'risk_factors': ['Regulatory/Legal Risk', 'Reputational Risk'],
            'mitigation_suggestions': ['Complete remediation required before reconsideration']
        })
        return decision_result
    
    # ═══════════════════════════════════════════════════════════════
    # TIER 2: HIGH RISK THRESHOLDS (Requires executive approval)
    # ═══════════════════════════════════════════════════════════════
    tier2_flags = []
    risk_factors = []
    
    # Financial Risk Assessment (Altman Z-Score: <1.8 = high risk, >2.9 = safe)
    if metrics['fin_score'] < 1.8 and metrics['fin_score'] > 0:  # Changed threshold logic
        tier2_flags.append(f"Critical financial distress (Altman Z-Score: {metrics['fin_score']:.2f})")
        risk_factors.append('Financial Stability Risk')
    
    # Quality Risk Assessment
    if metrics['avg_ppm'] > 1500:
        tier2_flags.append(f"Severe quality issues ({metrics['avg_ppm']:.0f} PPM defects)")
        risk_factors.append('Product Quality Risk')
    
    # Operational Risk Assessment
    if metrics['total_ncr'] > 8:
        tier2_flags.append(f"Excessive non-conformances ({metrics['total_ncr']} NCRs)")
        risk_factors.append('Operational Risk')
    
    if metrics['avg_otd'] < 80:
        tier2_flags.append(f"Poor delivery performance ({metrics['avg_otd']:.1f}% OTD)")
        risk_factors.append('Supply Chain Risk')
    
    # Geopolitical Risk Assessment
    if metrics['geo_score'] > 80:
        tier2_flags.append(f"High geopolitical risk (score: {metrics['geo_score']}/100)")
        risk_factors.append('Geopolitical Risk')
    
    # ESG Risk Assessment
    if metrics['esg_score'] > 70:
        tier2_flags.append(f"Significant ESG concerns (score: {metrics['esg_score']}/100)")
        risk_factors.append('ESG/Reputational Risk')
    
    # Compliance Risk Assessment - Updated to handle text values
    compliance_issues = []
    if metrics['acb'] in ['Non-Compliant', 'Under Investigation']:
        compliance_issues.append(f"Anti-bribery compliance: {metrics['acb']}")
    if metrics['tcomp'] in ['Non-Compliant', 'Under Investigation']:
        compliance_issues.append(f"Trade compliance: {metrics['tcomp']}")
    
    if compliance_issues:
        tier2_flags.extend(compliance_issues)
        risk_factors.append('Compliance Risk')
    
    # If multiple Tier 2 flags, automatic rejection
    if len(tier2_flags) >= 3:
        decision_result.update({
            'final_decision': 'REJECT',
            'decision_tier': 'TIER_2_HIGH_RISK',
            'confidence_level': 'HIGH',
            'reasoning': tier2_flags,
            'risk_factors': risk_factors,
            'mitigation_suggestions': [
                'Comprehensive supplier improvement plan required',
                'Third-party risk assessment recommended',
                'Executive committee review mandatory'
            ]
        })
        return decision_result
    
    # ═══════════════════════════════════════════════════════════════
    # TIER 3: MODERATE RISK - AI SYNTHESIS REQUIRED
    # ═══════════════════════════════════════════════════════════════
    tier3_flags = []
    
    # Moderate thresholds - Updated for Altman Z-Score
    if 1.8 <= metrics['fin_score'] < 2.9 and metrics['fin_score'] > 0:  # Changed threshold
        tier3_flags.append(f"Moderate financial concerns (Altman Z-Score: {metrics['fin_score']:.2f})")
    if 1000 < metrics['avg_ppm'] <= 1500:
        tier3_flags.append(f"Quality concerns ({metrics['avg_ppm']:.0f} PPM)")
    if 3 <= metrics['total_ncr'] <= 8:
        tier3_flags.append(f"Multiple non-conformances ({metrics['total_ncr']} NCRs)")
    if 1 <= metrics['open_majors'] < 5:
        tier3_flags.append(f"Audit findings pending ({metrics['open_majors']} major)")
    if 80 <= metrics['avg_otd'] < 90:
        tier3_flags.append(f"Delivery performance below target ({metrics['avg_otd']:.1f}%)")
    
    # Parse AI decision for nuanced analysis
    ai_decision = "UNKNOWN"
    ai_confidence = "LOW"
    
    try:
        import json
        import re
        
        # Extract JSON from AI response
        json_match = re.search(r'\{.*\}', gemini_decision, re.DOTALL)
        if json_match:
            ai_response = json.loads(json_match.group())
            ai_decision = ai_response.get('decision', 'UNKNOWN')
            ai_reasons = ai_response.get('reasons', [])
            
            # Assess AI confidence based on reasoning quality
            if len(ai_reasons) >= 3 and any('exceeds threshold' in reason for reason in ai_reasons):
                ai_confidence = "HIGH"
            elif len(ai_reasons) >= 2:
                ai_confidence = "MEDIUM"
    except:
        # Fallback to simple text parsing
        if "REJECT" in gemini_decision.upper():
            ai_decision = "REJECT"
        elif "APPROVE" in gemini_decision.upper():
            ai_decision = "APPROVE"
    
    # Decision logic for Tier 3
    if tier3_flags or tier2_flags:  # Any moderate/high risk flags
        if ai_decision == "REJECT":
            final_decision = "REJECT"
            confidence = "HIGH" if ai_confidence == "HIGH" else "MEDIUM"
            reasoning = tier2_flags + tier3_flags + ["AI analysis confirms rejection based on risk synthesis"]
        elif ai_decision == "APPROVE" and len(tier2_flags) == 0:
            final_decision = "CONDITIONAL_APPROVE"
            confidence = "MEDIUM"
            reasoning = tier3_flags + ["AI analysis suggests manageable risk with conditions"]
        else:
            final_decision = "REQUIRES_REVIEW"
            confidence = "LOW"
            reasoning = tier2_flags + tier3_flags + ["Conflicting risk signals require human review"]
    else:
        # Low risk scenario
        final_decision = "APPROVE" if ai_decision != "REJECT" else "CONDITIONAL_APPROVE"
        confidence = "HIGH"
        reasoning = ["Low risk profile confirmed by AI analysis"]
    
    # ═══════════════════════════════════════════════════════════════
    # MITIGATION SUGGESTIONS BASED ON RISK PROFILE
    # ═══════════════════════════════════════════════════════════════
    mitigation_suggestions = []
    
    if 'Financial Stability Risk' in risk_factors:
        mitigation_suggestions.extend([
            'Require financial guarantees or letters of credit',
            'Implement more frequent financial health monitoring',
            'Consider dual sourcing strategy'
        ])
    
    if 'Product Quality Risk' in risk_factors:
        mitigation_suggestions.extend([
            'Increase incoming inspection levels',
            'Implement supplier quality improvement program',
            'Require statistical process control implementation'
        ])
    
    if 'Supply Chain Risk' in risk_factors:
        mitigation_suggestions.extend([
            'Implement buffer inventory strategy',
            'Require delivery performance improvement plan',
            'Consider alternative suppliers for critical items'
        ])
    
    if 'Compliance Risk' in risk_factors:
        mitigation_suggestions.extend([
            'Require third-party compliance audit',
            'Implement enhanced due diligence monitoring',
            'Establish compliance milestone checkpoints'
        ])
    
    decision_result.update({
        'final_decision': final_decision,
        'decision_tier': 'TIER_3_MODERATE' if tier3_flags else 'TIER_4_LOW_RISK',
        'confidence_level': confidence,
        'reasoning': reasoning,
        'risk_factors': list(set(risk_factors)),
        'mitigation_suggestions': mitigation_suggestions,
        'review_required': final_decision in ['REQUIRES_REVIEW', 'CONDITIONAL_APPROVE']
    })
    
    return decision_result

def calculate_risk_score(metrics: dict) -> float:
    """Calculate a weighted risk score based on metrics."""
    weights = {
        'open_majors': 0.2,
        'avg_ppm': 0.15,
        'total_ncr': 0.15,
        'avg_otd': 0.1,
        'avg_dev': 0.1,
        'fin_score': 0.1,
        'geo_score': 0.1,
        'esg_score': 0.05,
    }
    score = 0
    score += weights['open_majors'] * (1 if metrics['open_majors'] > 0 else 0)
    score += weights['avg_ppm'] * (metrics['avg_ppm'] / 1000)
    score += weights['total_ncr'] * (metrics['total_ncr'] / 10)
    score += weights['avg_otd'] * (1 - metrics['avg_otd'] / 100)  # Higher OTD is better
    score += weights['avg_dev'] * (abs(metrics['avg_dev']) / 30)
    
    # Updated for Altman Z-Score (lower is worse, invert the logic)
    if metrics['fin_score'] > 0:
        score += weights['fin_score'] * max(0, (2.9 - metrics['fin_score']) / 2.9)
    else:
        score += weights['fin_score']  # Maximum penalty if no score
    
    score += weights['geo_score'] * (metrics['geo_score'] / 100)
    score += weights['esg_score'] * (metrics['esg_score'] / 100)
    return score

def run_assessment(supplier_id: str, need: str):
    # 1) Gather data
    metrics = fetch_structured(supplier_id)
    snippets = fetch_narratives(supplier_id, top_k=5)

    # 2) Build prompt
    prompt = build_prompt(metrics, snippets, supplier_id, need)

    # 3) Invoke Gemini
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        decision = resp.text
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        decision = None

    # 4) Calculate risk score
    risk_score = calculate_risk_score(metrics)
    decision_analysis = make_final_decision(metrics, decision, supplier_id)


    return {
        "supplier_id": supplier_id,
        "risk_score": risk_score,
        "decision_analysis": decision_analysis,
        "gemini_decision": decision,
        "metrics": metrics
    }

def main(supplier_id: str, need: str):
    """Main function to run the assessment."""
    logging.info(f"Running assessment for Supplier ID: {supplier_id} with Need: {need}")
    result = run_assessment(supplier_id, need)
    logging.info(f"Assessment Result: {result}")
    return result
    



