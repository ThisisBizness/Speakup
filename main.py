import os
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Request, Form, Depends, HTTPException, Header, Cookie
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

@app.post("/analyze")
async def analyze_video(
    request: Request, 
    video: UploadFile = File(...),
    session_id: str = Depends(get_or_create_session)
) -> Response:
    """
    Handles video upload, calls Gemini API for analysis, and returns feedback.
    Uses session management to store results for later reference.
    """
    logger.info(f"Received video file: {video.filename}, content type: {video.content_type}")

    # --- File Size Check --- 
    # Get file size. Need to read the file to know the size accurately with UploadFile.
    # This reads the entire file into memory first, which might be an issue for *very* large files
    # on memory-constrained systems. A streaming check would be more complex.
    try:
        # Seek to the end to get the size, then back to the start
        video.file.seek(0, os.SEEK_END)
        file_size = video.file.tell()
        video.file.seek(0)
        logger.info(f"Reported file size: {file_size} bytes")
        if file_size > MAX_UPLOAD_SIZE_BYTES:
            logger.warning(f"Upload failed: File size {file_size} exceeds limit of {MAX_UPLOAD_SIZE_BYTES} bytes.")
            return JSONResponse(
                status_code=413, # Payload Too Large
                content={"error": f"Video file is too large ({round(file_size / (1024*1024), 1)} MB). Maximum size is {MAX_UPLOAD_SIZE_MB} MB."}
            )
    except Exception as e:
        # If seeking fails, it might be a stream that doesn't support it.
        # Log a warning but proceed cautiously. A more robust solution might be needed.
        logger.warning(f"Could not determine file size before reading: {e}. Proceeding with upload attempt.")
        # Optionally, you could enforce a content-length header check here if available
    # --- End File Size Check ---

    if not video.content_type or not video.content_type.startswith('video/'): # Added check for None content_type
        logger.warning(f"Uploaded file is not a video or content type is missing: {video.content_type}")
        return JSONResponse(status_code=400, content={"error": "Uploaded file must be a video."})

    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API key not configured. Please check server setup."})

    # Save the uploaded file temporarily
    try:
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as temp_file:
            # Read the uploaded file in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await video.read(chunk_size):
                temp_file.write(chunk)
            
            temp_file_path = temp_file.name
            logger.info(f"Video saved temporarily at {temp_file_path}")
        
        # Initialize the model
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            logger.info(f"Model {MODEL_NAME} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            return JSONResponse(
                status_code=500, 
                content={"error": f"Failed to initialize AI model: {str(e)}"}
            )

        # Create the prompt
        prompt = create_analysis_prompt()
        # logger.info("Created analysis prompt") # Less verbose logging

        # Upload the file to Google AI
        google_file = None # Initialize google_file
        try:
            # Ensure temp_file_path exists before proceeding
            if not Path(temp_file_path).is_file():
                 raise FileNotFoundError(f"Temporary video file not found at {temp_file_path}")
            
            # Upload the video to Google's servers using the file path
            logger.info(f"Uploading video file from path: {temp_file_path}")
            google_file = genai.upload_file(
                path=temp_file_path, 
                mime_type=video.content_type,
                display_name=video.filename # Add display name
            )
            
            if not google_file:
                 raise Exception("Google AI file upload returned None.")
                 
            logger.info(f"Video uploaded successfully. File Name: {google_file.name}") # Use google_file.name
            
            # Call the model with the prompt and video
            try:
                # Define generation and safety settings before the API call attempts
                generation_config = {
                    "temperature": 0.2,  # Lower temperature for more focused/precise responses
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192, # Increased max output tokens
                }
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
                
                # --- Wait for file to become ACTIVE ---
                processing_timeout_seconds = GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS
                start_time = time.time()
                while True:
                    # Check file state
                    state = google_file.state
                    if state == 1:  # 1 means ACTIVE
                        break
                    elif state == 2:  # 2 means FAILED
                        error_msg = f"Google AI file upload failed with state: {state}"
                        logger.error(error_msg)
                        raise google_exceptions.FailedPrecondition(error_msg)
                    elif time.time() - start_time > processing_timeout_seconds:
                        error_msg = f"Google AI file upload timed out after {processing_timeout_seconds} seconds"
                        logger.error(error_msg)
                        raise google_exceptions.FailedPrecondition(error_msg)
                    # If state is not 1 (ACTIVE) or 2 (FAILED), keep waiting
                    await asyncio.sleep(GEMINI_FILE_POLL_INTERVAL_SECONDS)
                # ----------------------------------------
                
                # Add a small delay AFTER confirming ACTIVE state to mitigate potential race condition
                logger.info("Adding short delay before using the file...")
                await asyncio.sleep(GEMINI_FILE_RETRY_DELAY_SECONDS)

                logger.info("Calling Gemini API for video analysis (Attempt 1)...")
                try:
                    response = model.generate_content(
                        [prompt, google_file],
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        request_options={"timeout": GEMINI_REQUEST_TIMEOUT_SECONDS}, 
                        stream=False,
                    )
                except google_exceptions.FailedPrecondition as e:
                    # Specific check for the "not ACTIVE" error
                    if "not in an ACTIVE state" in str(e):
                        logger.warning(f"Gemini API reported file not ACTIVE on first attempt: {e}. Retrying after delay...")
                        await asyncio.sleep(5) # Wait 5 more seconds before retrying
                        logger.info("Calling Gemini API for video analysis (Attempt 2)...")
                        response = model.generate_content(
                            [prompt, google_file],
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                            request_options={"timeout": GEMINI_REQUEST_TIMEOUT_SECONDS}, 
                            stream=False,
                        )
                    else:
                        raise # Re-raise if it's a different FailedPrecondition error
                
                logger.info("Received response from Gemini API")

                # Check for blocked prompts or empty candidates
                if not response.candidates:
                     logger.warning("Gemini response has no candidates. Possible safety block or empty response.")
                     # Try to get block reason if available
                     block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', 'Unknown reason')
                     error_msg = f"AI analysis failed. The response was empty or blocked (Reason: {block_reason})."
                     raise google_exceptions.FailedPrecondition(error_msg)
                     
                response_text = response.text # Access text after checking candidates
                
                # Try to parse the response as JSON
                try:
                    # Clean potential markdown code fences
                    if response_text.strip().startswith("```") and response_text.strip().endswith("```"):
                        response_text = response_text.strip()[3:-3].strip() # Remove fences
                        # Remove potential language identifier (e.g., json)
                        if response_text.startswith("json\n"):
                             response_text = response_text[5:].strip()
                             
                    # Parse the JSON response
                    analysis_result = parse_gemini_response(response_text)
                    logger.info("Successfully parsed JSON response")
                    
                    # Store the analysis in the session
                    analysis_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "video_name": video.filename,
                        "analysis_id": str(uuid.uuid4()),
                        "result": analysis_result
                    }
                    SESSION_STORE[session_id]["video_analyses"].append(analysis_entry)
                    SESSION_STORE[session_id]["current_analysis_id"] = analysis_entry["analysis_id"]
                    SESSION_STORE[session_id]["conversation_history"] = [
                        {"role": "system", "content": "You are an AI speaking coach who has just analyzed a video. The user may ask follow-up questions about the analysis. Be helpful and specific in your answers."},
                        {"role": "user", "content": "I've uploaded a video of my speaking for analysis."},
                        {"role": "assistant", "content": f"I've analyzed your speaking performance. Here's my feedback: {json.dumps(analysis_result)}"}
                    ]
                    
                    # Add the session_id to the response for the client to store
                    analysis_result["session_id"] = session_id
                    analysis_result["analysis_id"] = analysis_entry["analysis_id"]
                    
                except (json.JSONDecodeError, ValueError) as json_err:
                    logger.error(f"Failed to parse AI response as JSON: {json_err}")
                    logger.debug(f"Raw AI response: {response.text}") # Log raw text on error
                    # Return the raw text with an error indicator
                    analysis_result = {
                        "error": "Failed to parse AI response as JSON.",
                        "raw_feedback": response.text, # Keep raw text separate
                        "scores": {},
                        "overall_score": 0,
                        "strengths": [],
                        "improvement_areas": ["Could not parse the AI response into the expected format."]
                    }
                    # Still save this partial result to session
                    analysis_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "video_name": video.filename,
                        "analysis_id": str(uuid.uuid4()),
                        "result": analysis_result
                    }
                    SESSION_STORE[session_id]["video_analyses"].append(analysis_entry)
                    SESSION_STORE[session_id]["current_analysis_id"] = analysis_entry["analysis_id"]
                    SESSION_STORE[session_id]["conversation_history"] = [
                         {"role": "system", "content": "You are an AI speaking coach. Analysis failed to parse correctly."},
                         {"role": "assistant", "content": f"I tried analyzing the video, but had trouble formatting the response. Here is the raw feedback: {response.text}"}
                     ]
                    analysis_result["session_id"] = session_id
                    analysis_result["analysis_id"] = analysis_entry["analysis_id"]

            except (google_exceptions.GoogleAPICallError, google_exceptions.RetryError, google_exceptions.FailedPrecondition) as api_err:
                logger.error(f"Error calling Gemini API: {api_err}")
                return JSONResponse(
                    status_code=500, 
                    content={"error": f"Error during AI analysis: {str(api_err)}"}
                )
            except Exception as e:
                 logger.exception("Unexpected error during Gemini API call") # Log full traceback
                 return JSONResponse(status_code=500, content={"error": "An unexpected error occurred during AI analysis."})
            finally:
                # Clean up the uploaded file on Google's side
                if google_file:
                    try:
                        logger.info(f"Attempting to delete Google AI file: {google_file.name}")
                        # genai.delete_file(google_file.name) # Use delete_file when available
                        # As of June 2024, direct deletion might still be unavailable/unreliable via SDK
                        # Files generally expire automatically after ~48 hours
                        logger.info("Note: Google AI file deletion via SDK might not be fully supported yet. Files typically expire automatically.")
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to delete Google AI uploaded file ({google_file.name}): {cleanup_err}")

        except Exception as upload_err:
            logger.error(f"Failed to upload or process video for Google AI: {upload_err}")
            return JSONResponse(
                status_code=500, 
                content={"error": f"Failed to prepare video for analysis: {str(upload_err)}"}
            )

    except Exception as e:
        logger.exception(f"Unexpected error processing video upload for {video.filename}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"An unexpected server error occurred while processing the video."}
        )
    finally:
        # Clean up the temporary file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} deleted")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temporary file: {cleanup_err}")

    logger.info(f"Returning analysis for {video.filename}")
    # Add Set-Cookie header to store the session ID
    response = JSONResponse(content=analysis_result)
    response.set_cookie(key="session_id", value=session_id, max_age=int(timedelta(hours=SESSION_TIMEOUT_HOURS).total_seconds()))
    return response

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
    logger.info(f"Received follow-up question: {question}")
    
    # Check if this session has any analysis
    if session_id not in SESSION_STORE or not SESSION_STORE[session_id].get("video_analyses"):
        return JSONResponse(
            status_code=400,
            content={"error": "No video analysis found in this session. Please upload a video first."}
        )
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Get conversation history
        conversation = SESSION_STORE[session_id]["conversation_history"]
        
        # Add the new question
        conversation.append({"role": "user", "content": question})
        
        # Create a formatted conversation for the model
        formatted_conversation = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in conversation
        ])
        
        # Call the model to get a response to the follow-up question
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024,
        }
        
        response = model.generate_content(
            formatted_conversation,
            generation_config=generation_config,
            stream=False
        )
        
        # Add the response to conversation history
        conversation.append({"role": "assistant", "content": response.text})
        SESSION_STORE[session_id]["conversation_history"] = conversation
        
        # Return the response
        return JSONResponse(content={"response": response.text})
        
    except Exception as e:
        logger.exception("Error processing follow-up question") # Log full traceback
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred while processing your question."}
        )

# Add a simple health check endpoint (optional but good practice)
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Check if API key is loaded before running
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Please create a .env file with your key.")
        print("Example .env file content:")
        print("GEMINI_API_KEY=YOUR_API_KEY_HERE")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000) 