# Speak Up - AI Speech Coach

Speak Up is an AI-powered platform that provides detailed feedback on speaking skills. Users can upload a video of themselves speaking (1-3 minutes), and the application uses Google's Gemini 2.5 Flash Preview model to analyze and provide actionable feedback on various aspects of public speaking.

## Features

- **Video Upload:** Support for various video formats up to 500MB
- **AI Analysis:** Detailed feedback on 8 key speaking dimensions:
  - Voice Clarity
  - Voice Tonality
  - Pacing
  - Body Language
  - Gestures
  - Eye Contact
  - Content Structure
  - Confidence
- **Actionable Insights:** Specific strengths and areas for improvement
- **Score Breakdown:** 1-10 rating for each speaking dimension

## Technical Stack

- **Backend:** FastAPI (Python)
- **AI:** Google's Gemini 2.5 Flash Preview model
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Ready for Vercel, AWS, or Heroku

## Setup Instructions

### Prerequisites

- Python 3.9+
- Google Generative AI API Key (for Gemini model access)

### Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/speak-up.git
cd speak-up
```

2. **Set up a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create a .env file with your API key**

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

5. **Run the application**

```bash
uvicorn main:app --reload
```

6. Open your browser and navigate to `http://127.0.0.1:8000`

### Deployment

#### Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project directory
3. Set the environment variable `GEMINI_API_KEY` in your Vercel project settings

#### Heroku

1. Create a Heroku app: `heroku create`
2. Set the API key: `heroku config:set GEMINI_API_KEY=your_api_key_here`
3. Deploy: `git push heroku main`

## Usage

1. Visit the application in your browser
2. Upload a video of yourself speaking (ideally 1-3 minutes in length)
3. Wait for the AI to analyze your speaking (this may take a minute or two)
4. Review your detailed feedback and score breakdown
5. Focus on the improvement areas to enhance your speaking skills

## Limitations

- Video size limit: 500MB
- Processing time depends on video length and complexity
- Internet connection required for video upload and AI processing
- The Gemini model works best with clearly visible speakers in good lighting
- Currently only supports English language analysis

## License

MIT License - See LICENSE file for details 