from fastapi import APIRouter, Request, Depends, HTTPException, Query
from .middlewares import with_authentication
from models.user import Role
from .helpers import sendError, sendSuccess
from typing import Dict, Literal
import requests
import json
import isodate
import os
from pydantic import BaseModel
from services.youtube_service import YouTubeService, VideoRequest
from typing import Literal, Dict
from models import JobMatch, ExternalJobMatch, Job, ExternalJob


router = APIRouter(
    prefix="/youtube",
    tags=["youtube"],
    dependencies=[
        Depends(with_authentication(
            [Role.JOB_SEEKER, Role.EMPLOYER, Role.ADMIN, Role.CREATOR, Role.ANALYST]))
    ],
)


youtube_service = YouTubeService()


@router.get("/videos")
async def get_videos_by_filter(
    request: Request,

    skill_type: Literal["user", "missing", "both"] = Query(
        default="user",
        description="Skills source: user skills, missing skills, or both"
    ),
    level: Literal["auto", "beginner", "intermediate", "advanced"] = Query(
        default="auto",
        description="Video level: auto-detect from match scores or fixed level"
    ),
    max_videos: int = Query(default=20, ge=1, le=100),
    max_videos_per_skill: int = Query(default=5, ge=1, le=20),
    skill=Query(default=None, description="Specific skill to search for")
):
    """Get YouTube videos based on user skills and match scores for intelligent level detection"""

    print('skill skill ', skill)

    video_request = VideoRequest(
        level=level,
        max_total_videos=max_videos,
        max_videos_per_skill=max_videos_per_skill,
        skill_type=skill_type,
        skill=skill
    )

    return await youtube_service.get_videos_internal(request.state.user["id"], video_request)
