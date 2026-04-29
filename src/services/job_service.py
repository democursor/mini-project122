"""
Background job service for async document processing
"""
import logging
import uuid
from datetime import datetime
from typing import Optional
from src.auth.supabase_db import get_db, supabase_error_handler

logger = logging.getLogger(__name__)


class JobService:
    """Manages background job status in Supabase"""
    
    def __init__(self):
        self.db = get_db()
    
    @supabase_error_handler
    def create_job(self, job_type: str = "document_processing") -> str:
        """Create a new job and return job_id"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "status": "pending",
            "job_type": job_type,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.db.table("jobs").insert(job_data).execute()
        logger.info(f"Created job: {job_id}")
        return job_id
    
    @supabase_error_handler
    def update_job_status(self, job_id: str, status: str, 
                         result: dict = None, error: str = None):
        """Update job status"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if result:
            update_data["result"] = result
        if error:
            update_data["error"] = error
        
        self.db.table("jobs").update(update_data).eq("id", job_id).execute()
        logger.info(f"Updated job {job_id}: {status}")
    
    @supabase_error_handler
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job status"""
        result = self.db.table("jobs").select("*").eq("id", job_id).single().execute()
        return result.data if result.data else None
