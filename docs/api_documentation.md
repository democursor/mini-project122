# API Documentation

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the API does not require authentication. Future versions will implement JWT-based authentication.

## Response Format

All API responses follow this structure:

**Success Response**:
```json
{
  "data": { ... },
  "status": "success"
}
```

**Error Response**:
```json
{
  "detail": "Error message",
  "status": "error"
}
```

## API Endpoints

### Documents API

#### Upload Document

Upload a PDF document for processing.

**Endpoint**: `POST /api/documents/upload`

**Request**:
- Conte