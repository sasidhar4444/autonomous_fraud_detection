"""
Automation Engine
Handles email alerts, webhooks, and Google Sheets integration
"""
import os
import math
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
from datetime import datetime as dt
import decimal
import pandas as pd
import numpy as np
import logging
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables
load_dotenv()

# Try to import gspread (optional)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

logger = logging.getLogger(__name__)

# def _make_json_serializable(obj):
#     """
#     Recursively convert common non-serializable types into JSON-safe types:
#     - pandas.Timestamp / datetime -> ISO string
#     - numpy types -> Python scalars
#     - decimal.Decimal -> float
#     - lists/arrays -> lists
#     - dicts -> sanitized dict
#     """
#     if obj is None or isinstance(obj, (str, bool, int, float)):
#         return obj

#     if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
#         try:
#             return obj.isoformat()
#         except Exception:
#             return str(obj)

#     if isinstance(obj, (np.integer, np.floating, np.bool_)):
#         return obj.item()

#     if isinstance(obj, decimal.Decimal):
#         try:
#             return float(obj)
#         except Exception:
#             return str(obj)

#     if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
#         return [_make_json_serializable(v) for v in list(obj)]

#     if isinstance(obj, dict):
#         return {str(k): _make_json_serializable(v) for k, v in obj.items()}

#     if hasattr(obj, "__dict__"):
#         return _make_json_serializable(obj.__dict__)

#     return str(obj)
def _make_json_serializable(obj):
    """
    Recursively convert obj into JSON-serializable types:
     - pd.Timestamp / datetime -> ISO string
     - pd.NaT, np.nan, inf, -inf -> None
     - numpy scalars -> native python scalars
     - decimal.Decimal -> float (or None if not finite)
     - pandas Series/ndarray/list/tuple/set -> list
     - dict -> sanitized dict (keys cast to str)
     - unknown objects -> str(obj) fallback
    """
    # None and primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj

    # pandas NaT
    if obj is pd.NaT:
        return None

    # pandas / python datetime -> ISO string
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    # numpy scalar -> python scalar
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj.item())
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj.item())
        if not math.isfinite(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj.item())

    # native float: check NaN/inf
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj

    # decimal
    if isinstance(obj, decimal.Decimal):
        try:
            f = float(obj)
            if not math.isfinite(f):
                return None
            return f
        except Exception:
            return str(obj)

    # lists / tuples / sets / numpy arrays / pandas Series -> list
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
        out = []
        for v in list(obj):
            out.append(_make_json_serializable(v))
        return out

    # dict -> sanitize keys and values
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            new[key] = _make_json_serializable(v)
        return new

    # Objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _make_json_serializable(vars(obj))
        except Exception:
            pass

    # fallback: try numeric conversion then string
    try:
        if isinstance(obj, (int,)):
            return int(obj)
        return str(obj)
    except Exception:
        return None

# Global flags to track if warnings have been logged
_email_config_warned = False
_webhook_config_warned = False

def _check_dry_run():
    """Check if DRY_RUN mode is enabled"""
    return os.getenv('DRY_RUN', '0') == '1'

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def send_email_alert(subject: str, body: Any, recipients: List[str]) -> Dict[str, Any]:
    """
    Send email via SMTP. Uses SMTP_* env vars.
    Returns result dict with success flag and details.
    Uses the _make_json_serializable helper if body is not a plain string.
    """
    DRY_RUN = _check_dry_run()
    
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    from_addr = smtp_user or f"noreply@{smtp_host}"

    # Prepare body text: if body is not a string, sanitize and pretty-print JSON
    if not isinstance(body, str):
        try:
            safe_body_obj = _make_json_serializable(body)
            # pretty JSON for email readability
            body_text = json.dumps(safe_body_obj, indent=2, ensure_ascii=False)
        except Exception:
            body_text = str(body)
    else:
        body_text = body

    # Compose raw email message
    msg = f"From: {from_addr}\r\nTo: {','.join(recipients)}\r\nSubject: {subject}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body_text}"

    if DRY_RUN:
        logger.info(f"Email prepared -> Subject: {subject} Recipients: {recipients}\nBody Preview: {body_text[:500]}")
        return {"success": True, "dry_run": True}

    # Validate SMTP configuration
    if not smtp_host or not smtp_user or not smtp_pass:
        logger.warning("Email credentials not configured")
        return {"success": False, "error": "smtp_not_configured"}

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, recipients, msg.encode("utf-8"))
        logger.info(f"Email sent -> subject={subject} recipients={recipients}")
        return {"success": True}
    except Exception as e:
        logger.exception("Email send failed")
        return {"success": False, "error": str(e)}

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
# def post_webhook(payload: Dict[str, Any], webhook_url: str = None) -> Dict[str, Any]:
#     """
#     POST JSON payload to webhook_url. Returns dict with success flag and status code.
#     This version sanitizes the payload so timestamps and pandas types won't break json encoding,
#     then logs a concise info about what was sent on success.
#     """
#     DRY_RUN = _check_dry_run()
    
#     url = webhook_url or os.getenv("WEBHOOK_URL")
#     if not url:
#         return {"success": False, "error": "no_webhook_configured"}

#     if DRY_RUN:
#         logger.info(f"Webhook prepared -> URL: {url} Payload keys: {list(payload.keys())}")
#         return {"success": True, "dry_run": True}

#     try:
#         # sanitize payload recursively
#         safe_payload = _make_json_serializable(payload)

#         # send as JSON (requests will encode it)
#         r = requests.post(url, json=safe_payload, timeout=10)

#         # Build concise info for logs: flags_count, sample_size, flags_full_path (if present)
#         info_payload = {
#             "flags_count": None,
#             "sample_size": 0,
#             "flags_full_path": None
#         }
#         try:
#             if isinstance(safe_payload, dict):
#                 info_payload["flags_count"] = safe_payload.get("flags_count")
#                 fs = safe_payload.get("flags_sample")
#                 if isinstance(fs, (list, tuple)):
#                     info_payload["sample_size"] = len(fs)
#                 meta = safe_payload.get("meta")
#                 if isinstance(meta, dict):
#                     info_payload["flags_full_path"] = meta.get("flags_full_path")
#         except Exception:
#             # swallow inspection errors but keep send success/failure clear
#             pass

#         logger.info(f"Webhook sent: status_code={r.status_code} info={info_payload}")

#         return {"success": r.ok, "status_code": r.status_code, "text": r.text, "info": info_payload}
#     except Exception as e:
#         logger.exception("Webhook POST failed")
#         return {"success": False, "error": str(e)}
def post_webhook(payload: Dict[str, Any], webhook_url: str = None) -> Dict[str, Any]:
    """
    POST sanitized payload to webhook_url. Returns dict with result.
    Ensures NaN/inf/pandas types converted to JSON-safe values.
    """
    DRY_RUN = _check_dry_run()

    url = webhook_url or os.getenv("WEBHOOK_URL", "")
    if not url:
        logger.warning("post_webhook: no webhook URL configured")
        return {"success": False, "error": "no_webhook_configured"}

    # DRY run: just log and return early
    if DRY_RUN:
        try:
            safe_sample = None
            if isinstance(payload, dict):
                safe_sample = _make_json_serializable(payload.get("flags_sample", payload))
            logger.info(f"post_webhook: DRY_RUN enabled - URL={url} payload_keys={list(payload.keys()) if isinstance(payload, dict) else 'non-dict'}")
            return {"success": True, "dry_run": True, "sample": safe_sample}
        except Exception:
            logger.info("post_webhook: DRY_RUN enabled (no sample available)")
            return {"success": True, "dry_run": True}

    # Sanitize payload
    try:
        safe_payload = _make_json_serializable(payload)
    except Exception as e:
        logger.exception("post_webhook: failed to sanitize payload")
        return {"success": False, "error": f"sanitization_failed: {e}"}

    # send
    try:
        r = requests.post(url, json=safe_payload, timeout=15)
        r.raise_for_status()
        # concise info for logs
        info_payload = {"flags_count": None, "sample_size": 0, "flags_full_path": None}
        try:
            if isinstance(safe_payload, dict):
                info_payload["flags_count"] = safe_payload.get("flags_count")
                fs = safe_payload.get("flags_sample")
                if isinstance(fs, (list, tuple)):
                    info_payload["sample_size"] = len(fs)
                meta = safe_payload.get("meta")
                if isinstance(meta, dict):
                    info_payload["flags_full_path"] = meta.get("flags_full_path")
        except Exception:
            pass

        logger.info(f"post_webhook: sent webhook status_code={r.status_code} info={info_payload}")
        return {"success": True, "status_code": r.status_code, "info": info_payload, "text": r.text}
    except Exception as e:
        logger.exception("post_webhook: Webhook POST failed")
        return {"success": False, "error": str(e)}

def handle_flagged_rows(flagged_rows: List[Dict], run_meta: Dict) -> Dict:
    """
    Handle batched flagged rows with single email/webhook alert
    
    Args:
        flagged_rows: List of flagged transaction dictionaries
        run_meta: Metadata about the run (timestamp, model_path, threshold, total_flagged)
        
    Returns:
        Dict summarizing automation results
    """
    count = len(flagged_rows)
    
    if count == 0:
        logger.info("No flagged rows to process")
        return {"count": 0}
    
    # Compose single payload
    flags_sample = flagged_rows[:10]  # First 10 rows as sample
    payload = {
        "meta": run_meta,
        "flags_count": count,
        "flags_sample": flags_sample
    }
    
    results = {
        "count": count,
        "email": None,
        "webhook": None
    }
    
    # Send email alert if configured
    alert_emails_str = os.getenv('ALERT_EMAILS', '')
    if not alert_emails_str:
        # Fallback to ALERT_EMAIL for backward compatibility
        alert_emails_str = os.getenv('ALERT_EMAIL', '')
    
    if alert_emails_str:
        recipients = [email.strip() for email in alert_emails_str.split(',') if email.strip()]
        if recipients:
            subject = f"[Alert] {count} flagged transactions"
            
            # Build email body
            body_lines = [
                f"Fraud Detection Alert",
                f"=" * 50,
                f"",
                f"Run Metadata:",
                json.dumps(run_meta, indent=2),
                f"",
                f"Total Flagged: {count}",
                f"",
                f"Sample Transactions (first 5):",
                f"-" * 50
            ]
            
            for i, row in enumerate(flags_sample[:5], 1):
                body_lines.append(f"\n{i}. Transaction:")
                body_lines.append(f"   - Timestamp: {row.get('timestamp', 'N/A')}")
                body_lines.append(f"   - User ID: {row.get('user_id', 'N/A')}")
                body_lines.append(f"   - Amount: ${row.get('amount', 'N/A')}")
                body_lines.append(f"   - Merchant: {row.get('merchant', 'N/A')}")
                body_lines.append(f"   - Method: {row.get('method', 'N/A')}")
                body_lines.append(f"   - Country: {row.get('country', 'N/A')}")
                body_lines.append(f"   - Probability: {row.get('probability', 'N/A'):.2%}" if 'probability' in row else f"   - Probability: N/A")
            
            if count > 5:
                body_lines.append(f"\n... and {count - 5} more flagged transactions")
            
            body = "\n".join(body_lines)
            
            try:
                results["email"] = send_email_alert(subject, body, recipients)
            except Exception as e:
                logger.error(f"Email alert failed after retries: {e}")
                results["email"] = {"success": False, "error": str(e)}
    
    # Send webhook if configured
    webhook_url = os.getenv('WEBHOOK_URL', '')
    if webhook_url:
        try:
            results["webhook"] = post_webhook(payload, webhook_url)
        except Exception as e:
            logger.error(f"Webhook failed after retries: {e}")
            results["webhook"] = {"success": False, "error": str(e)}
    
    return results

class AutomationEngine:
    """Handles automated actions based on predictions (legacy class for backward compatibility)"""
    
    def __init__(self):
        """Initialize automation engine with environment variables"""
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.alert_email = os.getenv('ALERT_EMAIL', '')
        self.webhook_url = os.getenv('WEBHOOK_URL', '')
        self.fraud_threshold = float(os.getenv('FRAUD_THRESHOLD', '0.7'))
        
        # Google Sheets setup
        self.gsheets_enabled = os.getenv('GSHEETS_ENABLED', 'false').lower() == 'true'
        self.gsheets_credentials_path = os.getenv('GSHEETS_CREDENTIALS_PATH', '')
        self.gsheets_sheet_name = os.getenv('GSHEETS_SHEET_NAME', 'Fraud Flags')
        
        if self.gsheets_enabled and GSPREAD_AVAILABLE:
            try:
                self._init_gsheets()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Sheets: {e}")
                self.gsheets_enabled = False
    
    def _init_gsheets(self):
        """Initialize Google Sheets client"""
        if not os.path.exists(self.gsheets_credentials_path):
            logger.warning("Google Sheets credentials file not found")
            self.gsheets_enabled = False
            return
        
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(
            self.gsheets_credentials_path, scopes=scope
        )
        self.gc = gspread.authorize(creds)
        logger.info("Google Sheets client initialized")
    
    def send_email_alert(self, transaction_data, probability):
        """
        Send email alert for flagged transaction (legacy method)
        
        Args:
            transaction_data: Dict with transaction details
            probability: Fraud probability
        """
        if not self.smtp_user or not self.smtp_password or not self.alert_email:
            logger.warning("Email credentials not configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email
            msg['Subject'] = f"Fraud Alert: Transaction Flagged (Probability: {probability:.2%})"
            
            body = f"""
            Fraud Alert Detected
            
            Transaction Details:
            - Timestamp: {transaction_data.get('timestamp', 'N/A')}
            - User ID: {transaction_data.get('user_id', 'N/A')}
            - Amount: ${transaction_data.get('amount', 'N/A')}
            - Merchant: {transaction_data.get('merchant', 'N/A')}
            - Method: {transaction_data.get('method', 'N/A')}
            - Country: {transaction_data.get('country', 'N/A')}
            - Fraud Probability: {probability:.2%}
            
            Please review this transaction immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for transaction {transaction_data.get('user_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_webhook(self, transaction_data, probability):
        """
        Send webhook POST request (legacy method)
        
        Args:
            transaction_data: Dict with transaction details
            probability: Fraud probability
        """
        if not self.webhook_url:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            payload = {
                'timestamp': dt.now().isoformat(),
                'transaction': transaction_data,
                'fraud_probability': probability,
                'flagged': True
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook sent successfully for transaction {transaction_data.get('user_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False
    
    def append_to_gsheets(self, transaction_data, probability):
        """
        Append flagged transaction to Google Sheets
        
        Args:
            transaction_data: Dict with transaction details
            probability: Fraud probability
        """
        if not self.gsheets_enabled:
            logger.warning("Google Sheets not enabled or not available")
            return False
        
        try:
            sheet = self.gc.open(self.gsheets_sheet_name).sheet1
            
            # Prepare row data
            row = [
                dt.now().isoformat(),
                transaction_data.get('timestamp', ''),
                transaction_data.get('user_id', ''),
                transaction_data.get('amount', ''),
                transaction_data.get('merchant', ''),
                transaction_data.get('method', ''),
                transaction_data.get('country', ''),
                f"{probability:.4f}"
            ]
            
            # Append row
            sheet.append_row(row)
            
            logger.info(f"Appended to Google Sheets: {transaction_data.get('user_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append to Google Sheets: {e}")
            return False
    
    def trigger_automation(self, transaction_data, probability):
        """
        Trigger all automation actions if probability exceeds threshold (legacy method)
        
        Args:
            transaction_data: Dict with transaction details
            probability: Fraud probability
            
        Returns:
            Dict with results of each automation action
        """
        if probability < self.fraud_threshold:
            return {'triggered': False, 'reason': 'Below threshold'}
        
        results = {
            'triggered': True,
            'probability': probability,
            'threshold': self.fraud_threshold,
            'email': False,
            'webhook': False,
            'gsheets': False
        }
        
        # Send email
        results['email'] = self.send_email_alert(transaction_data, probability)
        
        # Send webhook
        results['webhook'] = self.send_webhook(transaction_data, probability)
        
        # Append to Google Sheets
        if self.gsheets_enabled:
            results['gsheets'] = self.append_to_gsheets(transaction_data, probability)
        
        return results
