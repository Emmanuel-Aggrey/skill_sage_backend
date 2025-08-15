from fastapi import APIRouter, Request, Depends, HTTPException
from .middlewares import with_authentication
from models.user import Role
from .helpers import sendError, sendSuccess
from typing import List, Optional
import requests
import json
import isodate
import os
from pydantic import BaseModel

router = APIRouter(
    prefix="/youtube",
    tags=["youtube"],
    dependencies=[
        Depends(with_authentication([Role.JOB_SEEKER, Role.EMPLOYER, Role.ADMIN, Role.CREATOR, Role.ANALYST]))
    ],
)

class VideoRequest(BaseModel):
    skills: List[str]
    level: Optional[str] = "beginner"
    max_videos: Optional[int] = 10

def get_videos_for_skill(api_key: str, skill: str, level: str = "beginner", max_videos: int = 10):
    """
    Get real YouTube videos for a specific skill
    """
    # Construct search query based on skill and level
    if level == "beginner":
        query = f"{skill} tutorial for beginners learn {skill}"
    elif level == "intermediate":
        query = f"{skill} intermediate tutorial advanced {skill}"
    else:  # advanced
        query = f"{skill} advanced tutorial expert {skill} masterclass"

    # Step 1: Search for videos
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': min(max_videos, 50),
        'order': 'relevance',
        'videoDuration': 'medium',
        'videoDefinition': 'high',
        'relevanceLanguage': 'en',
        'key': api_key
    }

    try:
        # Make search request
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()

        if 'items' not in search_data or not search_data['items']:
            return {
                "error": "No videos found",
                "skill": skill,
                "level": level,
                "videos": []
            }

        # Extract video IDs
        video_ids = [item['id']['videoId'] for item in search_data['items']]

        # Step 2: Get detailed video information
        details_url = "https://www.googleapis.com/youtube/v3/videos"
        details_params = {
            'part': 'contentDetails,snippet,statistics',
            'id': ','.join(video_ids),
            'key': api_key
        }

        details_response = requests.get(details_url, params=details_params)
        details_response.raise_for_status()
        details_data = details_response.json()

        # Step 3: Format the results
        videos = []
        for item in details_data.get('items', []):
            # Convert duration from ISO 8601 to readable format
            duration_iso = item['contentDetails']['duration']
            duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds())

            # Format duration as MM:SS or HH:MM:SS
            if duration_seconds >= 3600:
                hours = duration_seconds // 3600
                minutes = (duration_seconds % 3600) // 60
                seconds = duration_seconds % 60
                duration_formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                minutes = duration_seconds // 60
                seconds = duration_seconds % 60
                duration_formatted = f"{minutes}:{seconds:02d}"

            # Clean description
            description = item['snippet']['description']
            if len(description) > 200:
                description = description[:200] + "..."

            video_info = {
                "id": item['id'],
                "title": item['snippet']['title'],
                "description": description,
                "thumbnail": item['snippet']['thumbnails']['high']['url'],
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "duration": duration_formatted,
                "view_count": int(item['statistics'].get('viewCount', 0)),
                "like_count": int(item['statistics'].get('likeCount', 0)),
                "published_date": item['snippet']['publishedAt'][:10],
                "channel_name": item['snippet']['channelTitle'],
                "skill": skill,
                "level": level
            }
            videos.append(video_info)

        # Sort by view count (most popular first)
        videos.sort(key=lambda x: x['view_count'], reverse=True)

        return {
            "skill": skill,
            "level": level,
            "total_results": len(videos),
            "videos": videos
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"API request failed: {str(e)}",
            "skill": skill,
            "level": level,
            "videos": []
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "skill": skill,
            "level": level,
            "videos": []
        }

@router.get("/videos/{skill}")
async def get_skill_videos(skill: str, level: str = "beginner", max_videos: int = 10):
    """
    Get YouTube videos for a specific skill
    """
    try:
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            return sendError("YouTube API key not configured")
        
        result = get_videos_for_skill(api_key, skill, level, max_videos)
        
        if "error" in result:
            return sendError(result["error"])
        
        return sendSuccess(result)
        
    except Exception as e:
        return sendError(f"Failed to fetch videos: {str(e)}")

@router.post("/videos/batch")
async def get_videos_for_skills(request: VideoRequest):
    """
    Get YouTube videos for multiple skills
    """
    try:
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            return sendError("YouTube API key not configured")
        
        all_videos = []
        
        for skill in request.skills:
            result = get_videos_for_skill(api_key, skill, request.level, request.max_videos)
            if "videos" in result and result["videos"]:
                all_videos.extend(result["videos"])
        
        # Sort all videos by view count
        all_videos.sort(key=lambda x: x['view_count'], reverse=True)
        
        return sendSuccess({
            "skills": request.skills,
            "level": request.level,
            "total_videos": len(all_videos),
            "videos": all_videos[:request.max_videos * len(request.skills)]
        })
        
    except Exception as e:
        return sendError(f"Failed to fetch videos: {str(e)}")

@router.get("/videos/recommended/{user_id}")
async def get_recommended_videos(request: Request, user_id: int, max_videos: int = 20):
    """
    Get YouTube videos based on user's skills and recommended skills
    """
    try:
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            return sendError("YouTube API key not configured")
        
        # This would integrate with your skills recommender
        # For now, we'll use some default popular skills
        popular_skills = ["python", "javascript", "react", "data science", "machine learning"]
        
        all_videos = []
        
        for skill in popular_skills[:3]:  # Limit to 3 skills to avoid API limits
            result = get_videos_for_skill(api_key, skill, "beginner", 5)
            if "videos" in result and result["videos"]:
                all_videos.extend(result["videos"])
        
        # Sort by view count and limit results
        all_videos.sort(key=lambda x: x['view_count'], reverse=True)
        
        return sendSuccess({
            "user_id": user_id,
            "total_videos": len(all_videos),
            "videos": all_videos[:max_videos]
        })
        
    except Exception as e:
        return sendError(f"Failed to fetch recommended videos: {str(e)}")
