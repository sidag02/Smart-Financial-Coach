"""Unusual transactions (FR-7): the service contract, reasons and candidate models.

scorer = load_service("unusual_transactions")
scorer.score_transactions(scoring_rows(txns, pool=txns))
# -> transaction_id, score, is_flagged, reason_code, evidence, model_version
"""
