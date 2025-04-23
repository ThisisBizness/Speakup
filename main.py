import os
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Request, Form, Depends, HTTPException, Header, Cookie, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import tempfile
import json
import base64
from pathlib import Path
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from google.api_core import exceptions as google_exceptions
import time
from pydantic import BaseModel # For request body validation

# --- Vercel Blob/KV Imports (Add these) ---
from vercel_blob import put as vercel_put, BlobClient, ClientError, HEADERS as BLOB_HEADERS
from vercel_kv import KVClient
import requests
# ----------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"
SESSION_TIMEOUT_HOURS = 24
MAX_UPLOAD_SIZE_MB = 500 # Max file size in MB
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
GEMINI_REQUEST_TIMEOUT_SECONDS = 300 # 5 minutes
GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS = 180 # 3 minutes
GEMINI_FILE_POLL_INTERVAL_SECONDS = 4
GEMINI_FILE_RETRY_DELAY_SECONDS = 5

# Session keys
SESSION_KEY_CREATED_AT = "created_at"
SESSION_KEY_LAST_ACTIVE = "last_active"
SESSION_KEY_ANALYSES = "video_analyses"
SESSION_KEY_CONVERSATION = "conversation_history"
SESSION_KEY_CURRENT_ANALYSIS_ID = "current_analysis_id"
SESSION_KEY_JOB_ID = "job_id"
SESSION_KEY_STATUS = "status"
SESSION_KEY_BLOB_URL = "blob_url"
SESSION_KEY_RESULT = "result"
SESSION_KEY_ERROR = "error"
JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_PROCESSING = "PROCESSING"
JOB_STATUS_COMPLETED = "COMPLETED"
JOB_STATUS_FAILED = "FAILED"

if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables.")
    # You might want to raise an exception or exit here in a real application
    # raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
else:
    try:
        genai.configure(api_key=API_KEY)
        logger.info("Google Generative AI SDK configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI SDK: {e}")
        # Handle configuration error appropriately

# Configure the Gemini model
# MODEL_NAME = "models/gemini-2.5-flash-preview-04-17"

# Simple in-memory session store (for demo/MVP purposes)
# WARNING: This is NOT suitable for production. Data is lost on restart.
# Use Redis, DynamoDB, or another persistent store for production.
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# Session management (simple implementation for MVP)
def get_or_create_session(session_id: Optional[str] = Cookie(None)) -> str:
    """Get existing session or create a new one"""
    if not session_id or session_id not in SESSION_STORE:
        # Create new session
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            SESSION_KEY_CREATED_AT: datetime.now(),
            SESSION_KEY_LAST_ACTIVE: datetime.now(),
            SESSION_KEY_ANALYSES: [],  # Store history of analyses
            SESSION_KEY_CONVERSATION: []  # For follow-up questions
        }
    else:
        # Update last active time
        SESSION_STORE[session_id][SESSION_KEY_LAST_ACTIVE] = datetime.now()
    
    # Clean up old sessions (older than SESSION_TIMEOUT_HOURS)
    cleanup_old_sessions()
    
    return session_id

def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT_HOURS"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session_data in list(SESSION_STORE.items()): # Iterate over a copy
        if current_time - session_data.get(SESSION_KEY_LAST_ACTIVE, current_time) > timedelta(hours=SESSION_TIMEOUT_HOURS):
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        if session_id in SESSION_STORE:
            del SESSION_STORE[session_id]
            logger.info(f"Cleaned up old session: {session_id}")

# Set up the FastAPI app with larger file size limits
app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory (optional, if you have CSS/JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

def create_analysis_prompt():
    """Create the detailed prompt for video analysis."""
    return """You are a professional speaking coach analyzing a video of someone speaking. You must provide critical, honest feedback on multiple aspects of their public speaking performance.

TASK:
Watch the video carefully and analyze the person's speaking performance. Focus on voice quality, body language, content structure, and overall delivery.

ANALYSIS CRITERIA:
1. Voice Clarity (1-10): Evaluate how clearly words are pronounced, if speech is easily understood. Consider accent, articulation, and enunciation.
2. Voice Tonality (1-10): Assess variation in tone, pitch, and emphasis. Is it engaging or monotonous? Does the speaker use their voice effectively to emphasize key points?
3. Pacing (1-10): Evaluate speaking speed, appropriate pausing, and rhythm. Is it too fast, too slow, or inconsistent?
4. Body Language (1-10): Analyze posture, stance, and overall physical presence. Do they appear confident and natural?
5. Gestures (1-10): Assess hand movements and physical expressions. Are they natural, distracting, or enhancing the message?
6. Eye Contact (1-10): Evaluate connection with the audience/camera. Do they maintain appropriate eye contact or appear disengaged?
7. Content Structure (1-10): Rate clarity of message, logical flow, and organization. Is the content well-structured with clear points?
8. Confidence (1-10): Assess overall perceived confidence and authority. Does the speaker seem self-assured and credible?

For each criterion, follow this scoring guideline:
- 1-3: Poor/Needs significant improvement
- 4-5: Below average
- 6-7: Good/Average
- 8-9: Very good/Above average
- 10: Excellent/Professional level

IMPORTANT: Be brutally honest and critical. Don't sugar-coat feedback. The goal is improvement through direct criticism. Point out specific moments/timestamps if possible.

RESPONSE FORMAT:
Respond in the following JSON format EXACTLY with no additional text before or after:

```json
{
  "feedback": "Overall comprehensive assessment in 2-3 detailed paragraphs. Be specific, critical, and direct about strengths and weaknesses.",
  "scores": {
    "Voice Clarity": 0-10,
    "Voice Tonality": 0-10,
    "Pacing": 0-10,
    "Body Language": 0-10,
    "Gestures": 0-10,
    "Eye Contact": 0-10,
    "Content Structure": 0-10,
    "Confidence": 0-10
  },
  "overall_score": 0-10,
  "strengths": [
    "Specific strength 1 with brief explanation",
    "Specific strength 2 with brief explanation",
    "Specific strength 3 with brief explanation"
  ],
  "improvement_areas": [
    "Specific improvement area 1 with detailed critical feedback",
    "Specific improvement area 2 with detailed critical feedback",
    "Specific improvement area 3 with detailed critical feedback",
    "Specific improvement area 4 with detailed critical feedback"
  ]
}
```

Remember to be very critical and demanding in your assessment. Don't be afraid to give low scores where deserved. Your feedback should push the speaker to significantly improve.

CRITICAL: Your response MUST be valid JSON that can be parsed programmatically. Do not include any natural language text outside the JSON structure. Begin with ```json and end with ```. Check that all JSON syntax is valid.
"""

# Update the parse_gemini_response function to better handle JSON extraction
def parse_gemini_response(response_text):
    """
    Parse the response from Gemini into a structured format, handling various edge cases.
    
    Args:
        response_text: The raw text response from the Gemini API
    
    Returns:
        dict: The parsed JSON data or a dict with error details
    """
    if not response_text or not isinstance(response_text, str):
        return {
            "error": "Empty or invalid response from AI",
            "raw_feedback": str(response_text)
        }
    
    # Try to extract JSON from the response
    json_content = response_text.strip()
    
    # Case 1: Response is wrapped in markdown code blocks
    if json_content.startswith("```") and "```" in json_content[3:]:
        # Extract content between code fences
        try:
            start_idx = json_content.find("```") + 3
            # Skip language identifier if present (like ```json)
            if "\n" in json_content[start_idx:]:
                start_idx = json_content.find("\n", start_idx) + 1
            end_idx = json_content.rfind("```")
            json_content = json_content[start_idx:end_idx].strip()
        except Exception:
            # If extraction fails, use the original text
            pass
    
    # Case 2: Detect if there's leading or trailing natural language text
    try:
        # Try to find JSON object markers
        first_brace = json_content.find("{")
        last_brace = json_content.rfind("}")
        
        if first_brace >= 0 and last_brace > first_brace:
            # Extract just the JSON part
            json_content = json_content[first_brace:last_brace+1]
    except Exception:
        # If extraction fails, use the current text
        pass
    
    # Try to parse the JSON
    try:
        data = json.loads(json_content)
        
        # Validate required fields
        if not isinstance(data, dict):
            return {
                "error": "AI response contains invalid scores data",
                "raw_feedback": response_text,
                **{k: v for k, v in data.items() if k != "scores"}
            }

        # Validate scores
        if not isinstance(data.get("scores"), dict):
            return {
                "error": "AI response contains invalid scores data",
                "raw_feedback": response_text,
                **{k: v for k, v in data.items() if k != "scores"}
            }
        
        # Validate overall_score
        if not isinstance(data.get("overall_score"), (int, float)):
            return {
                "error": "AI response contains invalid overall_score data",
                "raw_feedback": response_text,
                **{k: v for k, v in data.items() if k != "overall_score"}
            }
        
        # Validate strengths
        if not isinstance(data.get("strengths"), list):
            return {
                "error": "AI response contains invalid strengths data",
                "raw_feedback": response_text,
                **{k: v for k, v in data.items() if k != "strengths"}
            }
        
        # Validate improvement_areas
        if not isinstance(data.get("improvement_areas"), list):
            return {
                "error": "AI response contains invalid improvement_areas data",
                "raw_feedback": response_text,
                **{k: v for k, v in data.items() if k != "improvement_areas"}
            }
        
        # Ensure the expected fields are present and have correct types
        required_fields = {
            "feedback": str,
            "scores": dict,
            "overall_score": (int, float),
            "strengths": list,
            "improvement_areas": list
        }
        
        validated_data = {}
        missing_or_invalid = []

        for field, expected_type in required_fields.items():
            value = data.get(field)
            if value is None:
                 missing_or_invalid.append(f"{field} (missing)")
            elif not isinstance(value, expected_type):
                 missing_or_invalid.append(f"{field} (invalid type: expected {expected_type.__name__}, got {type(value).__name__})")
            else:
                 validated_data[field] = value
        
        if missing_or_invalid:
            error_msg = f"AI response is missing or has invalid fields: {'; '.join(missing_or_invalid)}"
            logger.warning(error_msg)
            # Return error but include any valid data found
            return {
                "error": error_msg,
                "raw_feedback": response_text,
                **validated_data # Include partially validated data
            }
        
        # Validate score values (ensure keys exist and values are numeric 0-10)
        # Define expected score keys if necessary, or validate dynamically
        expected_score_keys = [
            "Voice Clarity", "Voice Tonality", "Pacing", "Body Language", 
            "Gestures", "Eye Contact", "Content Structure", "Confidence"
        ]
        valid_scores = {}
        invalid_scores = []
        for key in expected_score_keys:
            value = validated_data["scores"].get(key)
            if value is None:
                 invalid_scores.append(f"{key} (missing)")
                 valid_scores[key] = 0 # Default missing score
                 continue
            try:
                score = float(value)
                valid_scores[key] = max(0.0, min(10.0, score))  # Clamp to 0.0-10.0
            except (ValueError, TypeError):
                invalid_scores.append(f"{key} (invalid value: '{value}')")
                valid_scores[key] = 0.0 # Default invalid score
        
        validated_data["scores"] = valid_scores
        if invalid_scores:
             logger.warning(f"AI response contained invalid scores: {'; '.join(invalid_scores)}")
             # Optionally add a warning to the data returned to the user
             validated_data["warning"] = f"Some scores were missing or invalid and defaulted to 0: {'; '.join(invalid_scores)}"
        
        # Validate overall score (ensure numeric 0-10)
        try:
            overall = float(validated_data["overall_score"])
            validated_data["overall_score"] = max(0.0, min(10.0, overall)) # Clamp
        except (ValueError, TypeError):
            logger.warning(f"Invalid overall_score '{validated_data['overall_score']}'. Recalculating average.")
            # Recalculate if invalid or missing
            num_scores = len(valid_scores)
            if num_scores > 0:
                 validated_data["overall_score"] = round(sum(valid_scores.values()) / num_scores, 1)
            else:
                 validated_data["overall_score"] = 0.0 # Default if no valid scores exist
        
        # Ensure strengths and improvement areas contain strings
        validated_data["strengths"] = [str(item) for item in validated_data.get("strengths", []) if item is not None]
        validated_data["improvement_areas"] = [str(item) for item in validated_data.get("improvement_areas", []) if item is not None]
        
        return validated_data
    except Exception as e:
        logger.error(f"Failed to parse AI response: {e}")
        return {
            "error": "Failed to parse AI response",
            "raw_feedback": response_text
        }

# --- Vercel Blob Upload Token Endpoint ---
class UploadTokenRequest(BaseModel):
    filename: str

@app.post("/api/upload-token", status_code=status.HTTP_200_OK)
async def create_upload_token(payload: UploadTokenRequest):
    """Generate a client-side upload token for Vercel Blob."""
    filename = payload.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    try:
        # Generate a token for client-side uploads
        # 'pathname' determines the path within the blob store
        # 'clientPayload' can be used to pass metadata (optional)
        # 'allowedContentTypes' restricts upload types
        # 'maximumSizeInBytes' restricts file size
        token_response = BlobClient().create_client_upload_token(
            pathname=f"uploads/{uuid.uuid4()}-{filename}", # Unique path
            allowed_content_types=["video/mp4", "video/quicktime", "video/webm", "video/ogg"], # Adjust as needed
            maximum_size_in_bytes=MAX_UPLOAD_SIZE_BYTES,
            # Add access: 'public' if blobs should be publicly accessible by URL
            # access='public', 
        )
        logger.info(f"Generated Vercel Blob upload token for: {filename}")
        return token_response
    except ClientError as e:
        logger.error(f"Vercel Blob ClientError generating token: {e.status_code} - {e.body}")
        raise HTTPException(status_code=500, detail=f"Could not get upload permission: {e.message}")
    except Exception as e:
        logger.exception("Unexpected error generating upload token")
        raise HTTPException(status_code=500, detail="Server error generating upload token.")
# --- End Vercel Blob Upload Token Endpoint ---

# --- Analysis Start Endpoint (replaces old /analyze) ---
class StartAnalysisRequest(BaseModel):
    blob_url: str
    filename: str
    content_type: str

@app.post("/api/start-analysis", status_code=status.HTTP_202_ACCEPTED)
async def start_analysis_job(payload: StartAnalysisRequest, session_id: str = Depends(get_or_create_session)):
    """Receives blob URL, queues analysis job in KV, returns job ID."""
    blob_url = payload.blob_url
    filename = payload.filename
    content_type = payload.content_type
    
    logger.info(f"Received request to start analysis for blob: {blob_url} (Session: {session_id})")

    if not kv_client:
        raise HTTPException(status_code=500, detail="KV Store not configured. Cannot queue job.")
    if not blob_url or not filename:
        raise HTTPException(status_code=400, detail="Missing blob_url or filename.")

    job_id = f"job_{uuid.uuid4()}"
    job_data = {
        SESSION_KEY_STATUS: JOB_STATUS_QUEUED,
        SESSION_KEY_BLOB_URL: blob_url,
        "filename": filename,
        "content_type": content_type,
        "session_id": session_id,
        "queued_at": datetime.now().isoformat(),
        SESSION_KEY_RESULT: None,
        SESSION_KEY_ERROR: None,
    }

    try:
        # Store job details in KV, set expiration (e.g., 24 hours)
        kv_client.set(job_id, json.dumps(job_data), ex=SESSION_TIMEOUT_HOURS * 3600)
        logger.info(f"Queued analysis job {job_id} for blob {blob_url}")
        
        # Optionally store job_id in user session if needed later
        SESSION_STORE[session_id][SESSION_KEY_JOB_ID] = job_id 
        
        return {SESSION_KEY_JOB_ID: job_id, SESSION_KEY_STATUS: JOB_STATUS_QUEUED}
    except Exception as e:
        logger.exception(f"Failed to queue job {job_id} in Vercel KV")
        raise HTTPException(status_code=500, detail="Failed to queue analysis job.")
# --- End Analysis Start Endpoint ---

# --- Analysis Status Endpoint ---
@app.get("/api/analysis-status/{job_id}", status_code=status.HTTP_200_OK)
async def get_analysis_status(job_id: str, session_id: str = Depends(get_or_create_session)):
    """Checks the status and result of an analysis job from KV."""
    logger.info(f"Checking status for job {job_id} (Session: {session_id})")
    if not kv_client:
        raise HTTPException(status_code=500, detail="KV Store not configured. Cannot check job status.")

    try:
        job_data_json = kv_client.get(job_id)
        if not job_data_json:
            logger.warning(f"Job {job_id} not found in KV.")
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        
        job_data = json.loads(job_data_json)
        
        # Basic check: ensure session requesting status matches job's session
        if job_data.get("session_id") != session_id:
             logger.warning(f"Session mismatch trying to access job {job_id}")
             raise HTTPException(status_code=403, detail="Forbidden")

        response_data = {
            SESSION_KEY_JOB_ID: job_id,
            SESSION_KEY_STATUS: job_data.get(SESSION_KEY_STATUS, "UNKNOWN"),
            SESSION_KEY_RESULT: job_data.get(SESSION_KEY_RESULT),
            SESSION_KEY_ERROR: job_data.get(SESSION_KEY_ERROR)
        }
        return response_data
        
    except json.JSONDecodeError:
         logger.error(f"Failed to decode job data for {job_id} from KV.")
         raise HTTPException(status_code=500, detail="Error retrieving job status.")
    except Exception as e:
        logger.exception(f"Failed to get status for job {job_id} from Vercel KV")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status.")
# --- End Analysis Status Endpoint ---

# --- Background Job Processor Endpoint (Triggered by Cron) ---
@app.post("/api/process-jobs", status_code=status.HTTP_200_OK)
async def process_queued_jobs(request: Request, x_vercel_cron_secret: Optional[str] = Header(None)):
    """Picks up QUEUED jobs from KV and processes them."""
    # --- Security Check (Important!) ---
    # Ensure this endpoint is secured, e.g., using a Vercel Cron secret
    expected_secret = os.getenv("VERCEL_CRON_SECRET")
    if not expected_secret or x_vercel_cron_secret != expected_secret:
        logger.warning("Unauthorized attempt to access /api/process-jobs")
        raise HTTPException(status_code=401, detail="Unauthorized")
    # --------------------------------

    logger.info("Cron job triggered: Processing queued analysis jobs...")
    if not kv_client:
        logger.error("KV client not available. Cannot process jobs.")
        return {"message": "KV client not available"}

    processed_count = 0
    failed_count = 0
    try:
        # Find queued jobs (Requires iterating keys - potentially slow for large KV stores)
        # WARNING: KV scan can be slow. Better patterns exist for large scale (e.g., separate queue list).
        queued_job_ids = []
        cursor = 0
        while True:
            cursor, keys = kv_client.scan(cursor=cursor, match="job_*")
            for key in keys:
                try:
                    job_data_json = kv_client.get(key)
                    if job_data_json:
                        job_data = json.loads(job_data_json)
                        if job_data.get(SESSION_KEY_STATUS) == JOB_STATUS_QUEUED:
                            queued_job_ids.append(key)
                except Exception:
                    logger.warning(f"Failed to check status for key {key} during scan", exc_info=True)
            if cursor == 0:
                break
        
        logger.info(f"Found {len(queued_job_ids)} queued jobs.")

        # Process one job per invocation (typical for serverless cron)
        if queued_job_ids:
            job_id_to_process = queued_job_ids[0] # Simple FIFO
            logger.info(f"Processing job: {job_id_to_process}")
            try:
                 # Mark as processing
                 job_data_json = kv_client.get(job_id_to_process)
                 if not job_data_json:
                      logger.warning(f"Job {job_id_to_process} disappeared before processing.")
                      return {"message": f"Job {job_id_to_process} not found"}
                 
                 job_data = json.loads(job_data_json)
                 job_data[SESSION_KEY_STATUS] = JOB_STATUS_PROCESSING
                 job_data["processing_started_at"] = datetime.now().isoformat()
                 kv_client.set(job_id_to_process, json.dumps(job_data), ex=SESSION_TIMEOUT_HOURS * 3600) # Update TTL
                 
                 # Perform the actual analysis
                 analysis_result = await perform_analysis(job_data)
                 
                 # Update job status with result or error
                 job_data[SESSION_KEY_STATUS] = JOB_STATUS_COMPLETED if "error" not in analysis_result else JOB_STATUS_FAILED
                 if "error" in analysis_result:
                     job_data[SESSION_KEY_ERROR] = analysis_result["error"]
                     job_data[SESSION_KEY_RESULT] = analysis_result.get("raw_feedback") # Store raw if error
                     failed_count += 1
                 else:
                     job_data[SESSION_KEY_RESULT] = analysis_result
                     processed_count += 1
                 
                 job_data["completed_at"] = datetime.now().isoformat()
                 kv_client.set(job_id_to_process, json.dumps(job_data), ex=SESSION_TIMEOUT_HOURS * 3600)
                 logger.info(f"Finished processing job {job_id_to_process}. Status: {job_data[SESSION_KEY_STATUS]}")

            except Exception as e:
                logger.exception(f"Failed processing job {job_id_to_process}")
                failed_count += 1
                try:
                    # Try to mark the job as failed in KV
                    job_data[SESSION_KEY_STATUS] = JOB_STATUS_FAILED
                    job_data[SESSION_KEY_ERROR] = f"Processing error: {str(e)}"
                    job_data["completed_at"] = datetime.now().isoformat()
                    kv_client.set(job_id_to_process, json.dumps(job_data), ex=SESSION_TIMEOUT_HOURS * 3600)
                except Exception as kv_err:
                     logger.error(f"Failed to update job {job_id_to_process} status to FAILED in KV: {kv_err}")
        else:
            logger.info("No queued jobs found to process.")

    except Exception as e:
        logger.exception("Error during job processing loop")
        return {"message": f"Error scanning/processing jobs: {str(e)}"}
        
    return {"message": f"Job processing finished. Processed: {processed_count}, Failed: {failed_count}"}

# --- Analysis Logic (Refactored for Background Task) ---
async def perform_analysis(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Downloads video, runs Gemini analysis, returns result/error dict."""
    blob_url = job_data.get(SESSION_KEY_BLOB_URL)
    filename = job_data.get("filename", "unknown_video")
    content_type = job_data.get("content_type", "video/mp4") # Default if missing
    job_id = job_data.get(SESSION_KEY_JOB_ID, "unknown_job")

    logger.info(f"Starting analysis for job {job_id}, blob: {blob_url}")

    # Download video from Blob URL
    temp_file_path = None
    google_file = None
    try:
        response = requests.get(blob_url, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        logger.info(f"Job {job_id}: Video downloaded temporarily to {temp_file_path}")

        # --- Existing Gemini Analysis Logic (Adapted) ---
        if not API_KEY:
            return {"error": "API key not configured on server."}
        
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = create_analysis_prompt()
        
        # Upload to Google AI
        logger.info(f"Job {job_id}: Uploading {temp_file_path} to Google AI...")
        google_file = genai.upload_file(
            path=temp_file_path,
            mime_type=content_type,
            display_name=filename
        )
        if not google_file:
             raise Exception("Google AI file upload returned None.")
        logger.info(f"Job {job_id}: Uploaded to Google AI. File Name: {google_file.name}")

        # Wait for Google file processing
        processing_start_time = time.time()
        while True:
            if time.time() - processing_start_time > GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS:
                 raise TimeoutError("Google AI file processing timed out.")
            file_state = genai.get_file(google_file.name)
            state_value = file_state.state
            logger.info(f"Job {job_id}: Checking Google file state {google_file.name}: {state_value}")
            if state_value == 1: break
            if state_value == 2: raise Exception("Google AI file processing failed.")
            await asyncio.sleep(GEMINI_FILE_POLL_INTERVAL_SECONDS)
        
        # Short delay before use
        await asyncio.sleep(GEMINI_FILE_RETRY_DELAY_SECONDS)
        
        # Call Gemini (with retry for 'not ACTIVE' error)
        analysis_result = {}
        try:
            logger.info(f"Job {job_id}: Calling Gemini API (Attempt 1)...")
            generation_config = { "temperature": 0.2, "top_p": 0.95, "top_k": 64, "max_output_tokens": 8192 }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            response = model.generate_content(
                [prompt, google_file],
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={"timeout": GEMINI_REQUEST_TIMEOUT_SECONDS},
                stream=False
            )
        except google_exceptions.FailedPrecondition as e:
            if "not in an ACTIVE state" in str(e):
                logger.warning(f"Job {job_id}: Gemini reported file not ACTIVE (Attempt 1). Retrying...")
                await asyncio.sleep(GEMINI_FILE_RETRY_DELAY_SECONDS)
                logger.info(f"Job {job_id}: Calling Gemini API (Attempt 2)...")
                response = model.generate_content([prompt, google_file], generation_config=generation_config, safety_settings=safety_settings, request_options={"timeout": GEMINI_REQUEST_TIMEOUT_SECONDS}, stream=False)
            else:
                raise
        
        logger.info(f"Job {job_id}: Received Gemini response.")
        if not response.candidates:
             block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', 'Unknown')
             raise Exception(f"AI analysis response empty/blocked (Reason: {block_reason}).")
        
        analysis_result = parse_gemini_response(response.text)
        # --- End Gemini Logic --- 

    except requests.exceptions.RequestException as e:
        logger.error(f"Job {job_id}: Failed to download video from Blob URL {blob_url}: {e}")
        return {"error": f"Failed to download video: {str(e)}"}
    except (google_exceptions.GoogleAPICallError, google_exceptions.RetryError, google_exceptions.FailedPrecondition) as e:
        logger.error(f"Job {job_id}: Google API error during analysis: {e}")
        return {"error": f"Google API error: {str(e)}"}
    except Exception as e:
        logger.exception(f"Job {job_id}: Unexpected error during analysis of {blob_url}")
        return {"error": f"Unexpected analysis error: {str(e)}"}
    finally:
        # Clean up temporary file
        if temp_file_path and Path(temp_file_path).exists():
            try:
                os.unlink(temp_file_path)
                logger.info(f"Job {job_id}: Cleaned up temp file {temp_file_path}")
            except Exception as cleanup_err:
                logger.warning(f"Job {job_id}: Failed to clean up temp file {temp_file_path}: {cleanup_err}")
        # Clean up Google AI file
        if google_file:
             try:
                 logger.info(f"Job {job_id}: Attempting to delete Google AI file: {google_file.name}")
                 # genai.delete_file(google_file.name) # Use when available
                 logger.info("Note: Google AI file deletion via SDK might not be fully supported yet.")
             except Exception as cleanup_err:
                 logger.warning(f"Job {job_id}: Failed to delete Google AI file {google_file.name}: {cleanup_err}")
                 
    return analysis_result
# --- End Analysis Logic ---

# Remove the old synchronous /analyze endpoint
# @app.post("/analyze") ... (Delete this old function)

# Update /ask endpoint to use new session keys
@app.post("/ask")
async def ask_followup_question(
    request: Request,
    question: str = Form(...),
    session_id: str = Depends(get_or_create_session)
) -> JSONResponse:
    """
    Handles follow-up questions about a previous video analysis.
    Uses the conversation history stored in the session.
    """
    logger.info(f"Received follow-up question for session {session_id}: '{question}'")
    
    # Check if this session has any analysis and conversation history
    if session_id not in SESSION_STORE or not SESSION_STORE[session_id].get(SESSION_KEY_CONVERSATION):
        logger.warning(f"No conversation history found for session {session_id}. Cannot process follow-up.")
        return JSONResponse(
            status_code=400,
            content={"error": "No previous analysis found in this session to ask about. Please upload a video first."}
        )
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Get conversation history
        conversation_history = SESSION_STORE[session_id][SESSION_KEY_CONVERSATION]
        
        # Add the new question
        conversation_history.append({"role": "user", "content": question})
        
        # Call the model to get a response to the follow-up question
        generation_config = { "temperature": 0.5, "top_p": 0.95, "top_k": 64, "max_output_tokens": 1024 }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]

        logger.info("Calling Gemini API for follow-up question...")
        response = model.generate_content(
            conversation_history,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False
        )
        logger.info("Received response from Gemini API for follow-up.")

        if not response.candidates:
             logger.warning("Follow-up response has no candidates.")
             block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', 'Unknown')
             error_msg = f"AI could not answer the question (Reason: {block_reason})."
             raise google_exceptions.FailedPrecondition(error_msg)

        response_text = response.text
        
        conversation_history.append({"role": "assistant", "content": response_text})
        SESSION_STORE[session_id][SESSION_KEY_CONVERSATION] = conversation_history # Update session
        
        return JSONResponse(content={"response": response_text})
        
    except (google_exceptions.GoogleAPICallError, google_exceptions.RetryError, google_exceptions.FailedPrecondition) as api_err:
        logger.error(f"Error calling Gemini API for follow-up: {api_err}")
        return JSONResponse(status_code=500, content={"error": f"Error getting response from AI: {str(api_err)}"})
    except Exception as e:
        logger.exception("Error processing follow-up question")
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred while processing your question."})

# Health check endpoint (remains the same)
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    return {"status": "ok"}

# Main execution block (remains the same)
if __name__ == "__main__":
    # Check if API key is loaded before running
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Please create a .env file with your key.")
        print("Example .env file content:")
        print("GEMINI_API_KEY=YOUR_API_KEY_HERE")
    else:
        # Initialize Vercel KV client
        try:
            kv_client = KVClient()
            logger.info("Vercel KV client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Vercel KV client: {e}. KV features will be disabled.")
            kv_client = None
        uvicorn.run(app, host="0.0.0.0", port=8000) 