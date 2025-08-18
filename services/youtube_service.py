import requests
import isodate
from typing import List, Dict, Any, Optional, Literal
import os
from services.enhanced_matching_system import JobCourseMatchingService
from routes.helpers import sendError, sendSuccess
from pydantic import BaseModel
from models.job import JobMatch, ExternalJob, ExternalJobMatch, Job
from db.connection import session
import html
from datetime import datetime


class VideoRequest(BaseModel):
    level: Literal["auto", "beginner", "intermediate", "advanced"] = "auto"
    max_videos_per_skill: int = 5
    max_total_videos: int = 20
    skill_type: Literal["user", "missing", "both"] = "user"
    skill: Optional[str] = None


class YouTubeService:
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key not configured")

    def get_videos_for_skill(self, skill: str, level: str = "beginner", max_videos: int = 10):
        query_map = {
            "beginner": f"{skill} tutorial for beginners",
            "intermediate": f"{skill} intermediate tutorial",
            "advanced": f"{skill} advanced tutorial expert"
        }
        query = query_map.get(level, query_map["beginner"])

        try:
            video_ids = self._search_videos(query, max_videos)
            if not video_ids:
                return {"error": "No videos found", "videos": []}

            videos = self._get_video_details(video_ids, skill, level)
            videos.sort(key=lambda x: x['view_count'], reverse=True)

            return {"skill": skill, "level": level, "videos": videos}
        except Exception as e:
            return {"error": str(e), "videos": []}

    async def get_videos_internal(self, user_id: int, video_request: VideoRequest):
        try:
            # Get skills to search
            if video_request.skill:
                skills = [video_request.skill]
            else:
                skills = self._get_skills_from_jobs(user_id, video_request)

            if not skills:
                return sendError("No skills found")

            # Get skill levels if auto
            skill_levels = {}
            if video_request.level == "auto":
                skill_levels = self._get_user_skill_levels(user_id)

            # Fetch videos
            all_videos = []
            for skill in skills:
                level = skill_levels.get(
                    skill, "beginner") if video_request.level == "auto" else video_request.level
                result = self.get_videos_for_skill(
                    skill, level, video_request.max_videos_per_skill)
                if result.get("videos"):
                    all_videos.extend(result["videos"])

            if not all_videos:
                return sendError("No videos found")

            # Sort and limit
            all_videos.sort(key=lambda x: x['view_count'], reverse=True)
            all_videos = all_videos[:video_request.max_total_videos]

            return sendSuccess({
                "skills": skills,
                "total_videos": len(all_videos),
                "videos": all_videos
            })

        except Exception as e:
            return sendError(f"Failed: {str(e)}")

    def _get_skills_from_jobs(self, user_id: int, video_request: VideoRequest):
        job_service = JobCourseMatchingService(session)
        job_recommendations = job_service.get_recommended_jobs(user_id=user_id)
        skills_data = job_service.extract_skills_from_job_recommendations(
            job_recommendations)

        if video_request.skill_type == "user":
            return skills_data.get("user_skills", [])
        elif video_request.skill_type == "missing":
            return skills_data.get("missing_skills", [])
        else:
            return skills_data.get("missing_skills", [])[:5] + skills_data.get("user_skills", [])[:5]

    def _get_user_skill_levels(self, user_id: int):
        try:
            skill_scores = {}

            # Get matches
            matches = session.query(JobMatch).filter(
                JobMatch.user_id == user_id).all()
            external_matches = session.query(ExternalJobMatch).filter(
                ExternalJobMatch.user_id == user_id).all()

            # Collect scores
            for match in matches + external_matches:
                if hasattr(match, 'matched_skills') and match.matched_skills:
                    for skill in match.matched_skills:
                        skill_scores.setdefault(
                            skill, []).append(match.match_score)

            # Determine levels
            skill_levels = {}
            for skill, scores in skill_scores.items():
                avg = sum(scores) / len(scores)
                if avg <= 40:
                    skill_levels[skill] = "beginner"
                elif avg <= 60:
                    skill_levels[skill] = "intermediate"
                else:
                    skill_levels[skill] = "advanced"

            return skill_levels
        except:
            return {}

    def _search_videos(self, query: str, max_results: int):
        response = requests.get("https://www.googleapis.com/youtube/v3/search", params={
            'part': 'snippet', 'q': query, 'type': 'video', 'maxResults': min(max_results, 50),
            'order': 'relevance', 'key': self.api_key
        })
        response.raise_for_status()
        data = response.json()
        return [item['id']['videoId'] for item in data.get('items', [])]

    def _get_video_details(self, video_ids: List[str], skill: str, level: str):
        response = requests.get("https://www.googleapis.com/youtube/v3/videos", params={
            'part': 'contentDetails,snippet,statistics', 'id': ','.join(video_ids), 'key': self.api_key
        })
        response.raise_for_status()

        videos = []
        for item in response.json().get('items', []):
            duration = int(isodate.parse_duration(
                item['contentDetails']['duration']).total_seconds())
            description = html.unescape(
                item['snippet']['description']).replace('\n', ' ')
            if len(description) > 200:
                description = description[:200] + "..."

            videos.append({
                "id": item['id'],
                "title": item['snippet']['title'],
                "description": description,
                "thumbnail": item['snippet']['thumbnails']['high']['url'],
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "duration": f"{duration//60}:{duration % 60:02d}",
                "view_count": int(item['statistics'].get('viewCount', 0)),
                "like_count": int(item['statistics'].get('likeCount', 0)),
                "published_date": self.readable_date(item['snippet']['publishedAt']),
                "channel_name": item['snippet']['channelTitle'],
                "skill": skill,
                "level": level
            })
        return videos

    def readable_date(self, date_str):
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime("%b %d, %Y")
