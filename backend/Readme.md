# Aloo Sahayak Backend API
Production-ready FastAPI backend for Potato Disease Assistant with RAG capabilities.
## Features
 **RESTful API** - Standard HTTP endpoints for chat management   **WebSocket Streaming** - Real-time message streaming   **SQLite Database** - Easy to migrate to PostgreSQL/MySQL   **Async Support** - High concurrency with async/await   **Comprehensive Logging** - Error tracking and debugging   **CORS Enabled** - Frontend integration ready   **API Documentation** - Auto-generated OpenAPI docs  
## Project Structure
```backend/├── main.py           # FastAPI application and endpoints├── config.py         # Configuration management├── schemas.py        # Pydantic models for request/response├── requirements.txt  # FastAPI dependencies├── .env.example      # Environment variables template└── __init__.py       # Package initialization```
## Installation
### Prerequisites- Python 3.8+- Virtual environment activated
### Step 1: Install Backend Dependencies
```powershellpip install -r backend/requirements.txt```
### Step 2: Install Core Dependencies (if not already installed)
```powershellpip install -r requirements.txt```
### Step 3: Configure Environment
Copy `.env.example` to `.env` in the root directory:
```powershellcopy backend\.env.example .env```
Update `.env` with your settings:
```DATABASE_TYPE=sqliteAPI_HOST=127.0.0.1API_PORT=8000DEBUG=FalseOPENAI_API_KEY=your_key_here```
## Running the Backend
### Development Mode (with auto-reload)
```powershellcd potato-llm-projectpython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload```
### Production Mode (no auto-reload)
```powershellpython -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4```
## API Endpoints
### Health Check
```GET /api/health```
**Response:**```json{  "status": "healthy",  "message": "API is running",  "database": "sqlite",  "api_version": "1.0.0"}```
### Chat Management
#### List All Chats```GET /api/chats```
**Response:**```json[  {    "id": "uuid-1234",    "name": "Potato Rust",    "created_at": "2026-02-08T10:30:45.123456",    "message_count": 5  }]```
#### Create New Chat```POST /api/chatsContent-Type: application/json
{  "name": "New Chat"}```
**Response:**```json{  "id": "uuid-5678",  "name": "New Chat",  "created_at": "2026-02-08T11:00:00.000000",  "message_count": 0}```
#### Get Chat Details```GET /api/chats/{chat_id}```
**Response:**```json{  "id": "uuid-1234",  "name": "Potato Rust",  "created_at": "2026-02-08T10:30:45.123456",  "messages": [    {      "id": 0,      "sender": "user",      "content": "What is potato rust?",      "timestamp": "2026-02-08T10:30:50.111111"    },    {      "id": 1,      "sender": "assistant",      "content": "Potato rust is...",      "timestamp": "2026-02-08T10:30:55.222222"    }  ]}```
#### Rename Chat```PUT /api/chats/{chat_id}Content-Type: application/json
{  "new_name": "Rust Prevention"}```
#### Delete Chat```DELETE /api/chats/{chat_id}```
**Response:**```json{  "status": "success",  "message": "Chat deleted"}```
### Messages
#### Get Chat Messages```GET /api/chats/{chat_id}/messages```
**Response:**```json[  {    "id": 0,    "sender": "user",    "content": "What is potato rust?",    "timestamp": "2026-02-08T10:30:50.111111"  },  {    "id": 1,    "sender": "assistant",    "content": "Potato rust is a fungal disease...",    "timestamp": "2026-02-08T10:30:55.222222"  }]```
#### Send Message (Non-Streaming)```POST /api/chats/{chat_id}/messagesContent-Type: application/json
{  "content": "How to prevent rust?",  "language": "English"}```
**Response:**```json{  "user_message": "How to prevent rust?",  "ai_response": "Prevention methods include...",  "sources_count": 3,  "language": "English"}```
### WebSocket Streaming
#### Stream Messages in Real-Time```WebSocket ws://127.0.0.1:8000/api/ws/chats/{chat_id}/stream```
**Send Message:**```json{  "message": "What are symptoms of late blight?",  "language": "English"}```
**Receive Chunks:**```json{  "type": "chunk",  "content": "Late blight"}```
**Complete Response:**```json{  "type": "complete",  "content": "Late blight is...",  "sources_count": 2}```
**Errors:**```json{  "type": "error",  "error": "Error message here"}```
## API Documentation
Once running, visit:
- **Interactive Docs**: http://127.0.0.1:8000/api/docs (Swagger UI)- **Alternative Docs**: http://127.0.0.1:8000/api/redoc (ReDoc)- **OpenAPI JSON**: http://127.0.0.1:8000/api/openapi.json
## Database Configuration
### SQLite (Default - Current)
No additional setup needed. Database file is created automatically at `chat_history.db`.
### PostgreSQL (Future Migration)
```powershell# Install PostgreSQL driverpip install psycopg2-binary
# Update .envDATABASE_TYPE=postgresqlDATABASE_URL=postgresql://user:password@localhost:5432/potato_db```
### MySQL (Future Migration)
```powershell# Install MySQL driverpip install pymysql
# Update .envDATABASE_TYPE=mysqlDATABASE_URL=mysql+pymysql://user:password@localhost:3306/potato_db```
## Testing the API
### Using curl
```bash# Health checkcurl http://127.0.0.1:8000/api/health
# List chatscurl http://127.0.0.1:8000/api/chats
# Create chatcurl -X POST http://127.0.0.1:8000/api/chats \  -H "Content-Type: application/json" \  -d '{"name":"Test Chat"}'```
### Using Python requests
```pythonimport requests
BASE_URL = "http://127.0.0.1:8000/api"
# Create chatresponse = requests.post(f"{BASE_URL}/chats", json={"name": "New Chat"})chat_id = response.json()["id"]
# Send messageresponse = requests.post(    f"{BASE_URL}/chats/{chat_id}/messages",    json={"content": "What is potato rust?", "language": "English"})
print(response.json())```
### Using WebSocket (Python)
```pythonimport asyncioimport websocketsimport json
async def stream_message():    uri = "ws://127.0.0.1:8000/api/ws/chats/your_chat_id/stream"        async with websockets.connect(uri) as websocket:        # Send message        await websocket.send(json.dumps({            "message": "What is late blight?",            "language": "English"        }))                # Receive streaming response        while True:            try:                response = await websocket.recv()                data = json.loads(response)                print(data)                                if data.get("type") == "complete":                    break            except:                break
asyncio.run(stream_message())```
## Performance Considerations
### Concurrency
FastAPI with Uvicorn handles many concurrent WebSocket connections:
```powershell# Run with multiple workers for productionpython -m uvicorn backend.main:app --workers 4```
### Database
SQLite handles up to several hundred concurrent connections. For higher scale:
- **< 100 concurrent users**: SQLite (current)- **100-1000 concurrent users**: PostgreSQL- **1000+ concurrent users**: PostgreSQL with read replicas
### Streaming
WebSocket streaming provides real-time responses. Non-streaming endpoints wait for full response.
## Troubleshooting
### Port Already in Use
```powershell# Find process using port 8000netstat -ano | findstr :8000
# Kill processtaskkill /PID <PID> /F```
### Database Errors
```powershell# Reset database (delete old file)Remove-Item chat_history.db -Force```
### Import Errors
Ensure the project root is in Python path:
```powershellset PYTHONPATH=%PYTHONPATH%;c:\path\to\potato-llm-projectpython -m uvicorn backend.main:app --reload```
## Migration to PostgreSQL/MySQL
When ready to scale:
1. Update `DATABASE_TYPE` in `.env`2. Update `DATABASE_URL` with connection string3. Install database driver: `pip install psycopg2-binary` or `pip install pymysql`4. The API automatically uses the configured database5. Export SQLite data if needed
## Next Steps
1. Test all endpoints using API docs2. Test WebSocket streaming3. Create frontend client (React/Streamlit)4. Deploy to production server5. Monitor performance and logs6. Migrate to PostgreSQL if needed
## Support
For issues or questions, check:- API docs: http://127.0.0.1:8000/api/docs- Main app logs in terminal- Database file: `chat_history.db`
