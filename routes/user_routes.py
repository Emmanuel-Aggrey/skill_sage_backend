
import email
from pydantic import BaseModel, EmailStr

from models.user import (
    User,
    Role,
    Education,
    JobSeeker,
    Experience,
    Skill,
    JobSeekerSkill,
    File,
    UserResume,
)
from models.job import Job, JobMatch, UserJobPreferences, ExternalJob, ExternalJobMatch, Course
from .helpers import sendError, sendSuccess, getSha
from .middlewares import with_authentication
from db.connection import session
from fastapi import APIRouter, Request, Depends, status, UploadFile, Response, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import datetime

import uuid
import copy
from fastapi.responses import JSONResponse
from models.job import Bookmark
from sqlalchemy import func
from services.enhanced_matching_system import GenericLLMProcessor, JobCourseMatchingService, UserProfile
from services.matching_cache_manager import (
    EnhancedMatchingController, MatchingSystemConfig, UserMatchingPreferences
)
from services.websocket_manager import ws_manager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect


def get_enhanced_controller(session) -> EnhancedMatchingController:
    """Dependency to get enhanced matching controller"""

    llm_processor = GenericLLMProcessor()
    return EnhancedMatchingController(session, llm_processor)


llm = GenericLLMProcessor()
controller = get_enhanced_controller(session)


def get_matching_service(session) -> JobCourseMatchingService:
    """Get matching service instance"""
    return JobCourseMatchingService(session)


router = APIRouter(
    prefix="/user",
    tags=["user"],
    dependencies=[
        Depends(
            with_authentication(
                [Role.JOB_SEEKER, Role.ADMIN, Role.ADMIN, Role.CREATOR])
        )
    ],
)

app_router = APIRouter(
    prefix="/user",
    tags=["user"],
)


@router.get("/")
async def get_user(request: Request):
    try:
        user_id = request.state.user["id"]

        user = session.query(User).join(
            User.profile).filter(User.id == user_id).first()
        if user is None:
            return sendError("user not found")
        education = session.query(Education).filter(
            Education.user_id == user_id).all()
        exp = session.query(Experience).filter(
            Experience.user_id == user_id).all()
        user.education = education
        user.experience = exp
        user.profile.id
        user.experiences
        user.skills
        user.education

        skills = []
        us = (
            session.query(JobSeekerSkill)
            .join(JobSeekerSkill.skill)
            .filter(JobSeekerSkill.user_id == user_id)
            .all()
        )
        for i in us:
            skills.append({"name": i.skill.name, "id": i.skill.id})

        base_url = request.url._url
        resume_links = []
        latest_llm_insights = None

        links = session.query(UserResume).filter(
            UserResume.user_id == user_id
        ).order_by(UserResume.id.desc()).all()  # Order by latest first

        bookmark_count = (
            session.query(func.count(Bookmark.id))
            .filter(Bookmark.user_id == user_id)
            .scalar()
        )

        for i in links:
            resume_links.append(base_url + "file/" + i.filename)
            if i.llm_insights:
                latest_llm_insights = i.llm_insights

        user.skills = skills
        user.resume = resume_links
        user.llm_insights = llm.parse_json_output(latest_llm_insights)
        user.bookmark_count = bookmark_count

        u = copy.copy(user)

        if user.profile_image is not None:
            img = user.profile_image
            u.profile_image = base_url + "file/" + img

        return sendSuccess(u.to_json())
    except Exception as err:
        session.rollback()
        return sendError(err.args)
    finally:
        session.close()


class UpdateUser(BaseModel):
    name: str
    email: EmailStr


class ExperienceData(BaseModel):
    id: Optional[int] = None
    company_name: str
    job_title: str
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    is_remote: bool = False
    has_completed: bool = False
    tasks: Optional[str] = None


MATCHING_CONFIG = MatchingSystemConfig.load_from_env()


@router.post("/upload_resume", status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile,
    request: Request,
    background_tasks: BackgroundTasks,
    auto_match: bool = Query(
        True, description="Automatically generate matches after upload"),
    match_threshold: float = Query(
        40.0, description="Minimum match score threshold")
):

    user_id = request.state.user["id"]

    try:
        new_file = await file.read()
        fileSha = getSha(new_file)
        ex_chunk = file.filename.split(".")
        ext = ex_chunk[-1]
        filename = str(uuid.uuid4()) + "." + ext

        # Create comprehensive user profile
        user_profile = llm.create_user_profile(user_id, new_file, 'pdf')

        # Extract additional insights
        career_insights = llm.query_llm_with_template(
            # Limit text for API efficiency
            text=user_profile.resume_text[:2000],
            custom_prompt="""
            Analyze this resume and provide insights in JSON format:
            {
                "career_stage": "entry/mid/senior/executive",
                "primary_domain": "main field/industry",
                "years_experience": estimated_years,
                "key_strengths": ["strength1", "strength2", "strength3"],
                "growth_areas": ["area1", "area2"],
                "recommended_roles": ["role1", "role2", "role3"]
            }
            """
        )

        try:
            insights = llm.parse_json_output(career_insights)
        except Exception as e:
            print(f"Error parsing JSON output: {e}")
            insights = {"career_stage": "unknown", "primary_domain": "unknown"}

        # Save skills (existing logic enhanced)
        saved_skills = []
        if user_profile.skills:
            for skill_name in user_profile.skills:
                if not isinstance(skill_name, str) or not skill_name.strip():
                    continue

                cleaned_name = skill_name.strip()
                lower_name = cleaned_name.lower()

                # Enhanced skill matching with synonyms
                existing_skill = (
                    session.query(Skill)
                    .filter(Skill.lower == lower_name)
                    .first()
                )

                if existing_skill is None:
                    new_skill = Skill(cleaned_name)
                    new_skill.lower = lower_name
                    session.add(new_skill)
                    session.flush()
                    skill_id = new_skill.id
                else:
                    skill_id = existing_skill.id

                # Create user-skill mapping
                existing_map = (
                    session.query(JobSeekerSkill)
                    .filter(
                        JobSeekerSkill.user_id == user_id,
                        JobSeekerSkill.skill_id == skill_id,
                    )
                    .first()
                )

                if not existing_map:
                    jss = JobSeekerSkill(user_id=user_id, skill_id=skill_id)
                    session.add(jss)
                    saved_skills.append(cleaned_name)

        # Handle existing resumes
        existing_resumes = (
            session.query(UserResume)
            .filter(UserResume.user_id == user_id)
            .all()
        )
        for er in existing_resumes:
            old_file = session.query(File).filter(
                File.filename == er.filename).first()
            if old_file:
                session.delete(old_file)
            session.delete(er)

        # Save new resume
        resume_file = File(
            data=new_file, filename=filename, type=file.content_type, sha=fileSha
        )
        session.add(resume_file)
        session.flush()

        # Enhanced resume record with insights
        user_resume = UserResume(
            filename=resume_file.filename,
            user_id=user_id,
            resume_text=user_profile.resume_text,
            llm_insights=insights
        )
        session.add(user_resume)

        # Update user matching preferences if this is first upload
        prefs = session.query(UserMatchingPreferences).filter(
            UserMatchingPreferences.user_id == user_id
        ).first()

        if not prefs:
            prefs = UserMatchingPreferences(
                user_id=user_id,
                min_match_score=match_threshold,
                enable_semantic_matching=True,
                auto_refresh_matches=True
            )
            session.add(prefs)

        session.commit()

        # Schedule background matching if enabled (cache will be cleared in background task)
        if auto_match:
            background_tasks.add_task(
                enhanced_background_matching,
                user_id,
                match_threshold,
                include_courses=True,
                force_refresh=True
            )

        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": f"{file.filename} uploaded successfully",
                "data": {
                    "skills_extracted": len(user_profile.skills),
                    "skills_saved": len(saved_skills),
                    "experience_level": user_profile.experience_level,
                    "career_insights": insights,
                    "auto_matching_scheduled": auto_match,
                    "match_threshold": match_threshold
                }
            }
        )

    except Exception as err:
        session.rollback()
        print(f"Enhanced resume upload error: {err}")
        return JSONResponse(
            status_code=500,
            content={"success": False,
                     "error": "Internal Server Error", "details": str(err)}
        )
    finally:
        session.close()


async def enhanced_background_matching(user_id: int, match_threshold: float = 40.0,
                                       include_courses: bool = True, force_refresh: bool = False):
    """Enhanced background task for generating matches with caching"""
    # Create a new session for the background task
    from db.connection import engine
    from sqlalchemy.orm import Session
    bg_session = Session(bind=engine, autoflush=False, autocommit=False)

    try:
        # Add a small delay to ensure the main transaction is committed
        import asyncio
        await asyncio.sleep(0.5)

        matching_service = JobCourseMatchingService(bg_session)
        controller = get_enhanced_controller(bg_session)

        # Clear any existing cache for this user
        controller.cache_manager.invalidate_user_cache(user_id)

        # Debug: Check if user profile exists
        user_profile = matching_service.get_user_profile_from_db(user_id)
        print(
            f"Background task - User {user_id} profile: {'Found' if user_profile else 'Not found'}")
        if user_profile:
            print(
                f"Background task - User {user_id} skills count: {len(user_profile.skills)}")
        else:
            print(
                f"Background task - Cannot proceed without user profile for user {user_id}")
            return

        # Process job matching
        job_result = controller.process_user_matching_request(
            user_id, 'job', force_refresh=force_refresh, limit=50
        )

        # Process external job matching
        # Debug: Check external jobs before matching
        from models.job import ExternalJob
        external_jobs = bg_session.query(ExternalJob).filter(
            ExternalJob.is_enabled == True).all()
        jobs_with_skills = [
            job for job in external_jobs if job.skills and len(job.skills) > 0]
        print(
            f"Background task - External jobs: {len(external_jobs)} total, {len(jobs_with_skills)} with skills")

        external_job_result = controller.process_user_matching_request(
            user_id, 'external_job', force_refresh=force_refresh, limit=50
        )

        # Debug logging to see the structure of results
        print(
            f"Debug: Job result keys: {job_result.keys() if job_result else 'None'}")
        print(
            f"Debug: External job result keys: {external_job_result.keys() if external_job_result else 'None'}")

        if job_result and 'recommendations' in job_result:
            print(
                f"Debug: Job recommendations count: {len(job_result['recommendations'])}")
            if job_result['recommendations']:
                print(
                    f"Debug: First job recommendation: {job_result['recommendations'][0]}")

        if external_job_result and 'recommendations' in external_job_result:
            print(
                f"Debug: External job recommendations count: {len(external_job_result['recommendations'])}")
            if external_job_result['recommendations']:
                print(
                    f"Debug: First external job recommendation: {external_job_result['recommendations'][0]}")

        # Process course matching if enabled
        course_result = None
        if include_courses:
            course_result = controller.process_user_matching_request(
                user_id, 'course', force_refresh=force_refresh, limit=20
            )

        for result in [job_result, external_job_result]:
            recommendations = result.get('recommendations', [])
            if recommendations:
                try:
                    # Get the type from the first item's item_type field
                    job_type = recommendations[0].get('item_type')
                    if job_type:
                        matches = convert_recommendations_to_match_results(
                            recommendations, job_type)
                        matching_service.save_job_matches(user_id, matches)
                    else:
                        print(
                            f"Warning: No item_type found in recommendations for user {user_id}")
                except Exception as e:
                    print(f"Error saving matches for user {user_id}: {e}")
                    continue

        print(f"Enhanced matching completed for user {user_id}:")
        print(f"  - Jobs: {len(job_result.get('recommendations', []))}")
        print(
            f"  - External Jobs: {len(external_job_result.get('recommendations', []))}")
        if course_result:
            print(
                f"  - Courses: {len(course_result.get('recommendations', []))}")

    except Exception as e:
        print(f"Enhanced background matching error for user {user_id}: {e}")
    finally:

        await ws_manager.send_user_notification(str(user_id),
                                                {"type": "jobs_updated",
                                                 "message": "Upload complete!"}
                                                )

        bg_session.close()


def convert_recommendations_to_match_results(recommendations: List[Dict], item_type: str) -> List:
    """Convert recommendation dictionaries back to MatchResult format for saving"""
    from services.enhanced_matching_system import MatchResult

    matches = []
    for rec in recommendations:
        match_result = MatchResult(
            item_id=rec.get('item_id'),  # Use 'item_id' from recommendation
            item_type=item_type,
            match_score=rec.get('match_score', 0),
            skill_match_count=rec.get('skill_match_count', 0),
            total_required_skills=rec.get('total_required_skills', 0),
            missing_skills=rec.get('missing_skills', []),
            matched_skills=rec.get('matched_skills', []),
            item_data=rec  # Pass the entire record as item_data
        )
        matches.append(match_result)

    return matches


@router.get("/me/recommendations")
async def get_user_recommendations(request: Request,  min_match_score: int = 40, limit: int = 20):
    """Get user's recommended jobs"""
    user_id = request.state.user["id"]

    try:
        controller = get_enhanced_controller(session)
        matching_service = JobCourseMatchingService(
            session, controller.llm_processor)

        recommended_job = matching_service.get_recommended_jobs(
            user_id=user_id, min_match_score=float(min_match_score), limit=limit)

        return JSONResponse(content={"success": True, "data": recommended_job})

    except Exception as err:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:
        session.close()


@router.get("/recommendations")
async def get_enhanced_recommendations(
    request: Request,
    item_type: str = Query(
        'job', description="Type: job, external_job, course"),
    min_match_score: Optional[float] = Query(
        None, description="Minimum match score override"),
    limit: Optional[int] = Query(20, description="Maximum recommendations"),
    force_refresh: Optional[bool] = Query(
        False, description="Force cache refresh"),
    include_insights: Optional[bool] = Query(
        True, description="Include matching insights"),
    sort_by: Optional[str] = Query(
        'match_score', description="Sort by: match_score, relevance, date")
):
    """
    Get enhanced recommendations with intelligent caching and insights
    """
    user_id = request.state.user["id"]

    try:
        controller = get_enhanced_controller(session)

        # Get user preferences to determine min_match_score if not provided
        if min_match_score is None:
            user_prefs = controller.optimized_service.get_user_preferences(
                user_id)
            min_match_score = user_prefs.get('min_match_score', 40.0)

        # Process matching request
        result = controller.process_user_matching_request(
            user_id, item_type, force_refresh, limit)

        recommendations = result.get('recommendations', [])

        # Filter by minimum score
        filtered_recommendations = [
            rec for rec in recommendations
            if rec.get('match_score', 0) >= min_match_score
        ]

        # Apply sorting
        if sort_by == 'match_score':
            filtered_recommendations.sort(
                key=lambda x: x.get('match_score', 0), reverse=True)
        elif sort_by == 'relevance':
            # Custom relevance scoring (match_score + skill_match_count)
            filtered_recommendations.sort(
                key=lambda x: (x.get('match_score', 0) +
                               x.get('skill_match_count', 0) * 2),
                reverse=True
            )

        # Add insights if requested
        if include_insights and filtered_recommendations:
            insights = await generate_recommendation_insights(
                user_id, filtered_recommendations[:5], item_type
            )
        else:
            insights = {}

        # Get user profile summary
        matching_service = JobCourseMatchingService(session)
        user_profile = matching_service.get_user_profile_from_db(user_id)

        response_data = {
            "recommendations": filtered_recommendations,
            "total_count": len(filtered_recommendations),
            "insights": insights,
            "user_context": {
                "skills_count": len(user_profile.skills) if user_profile else 0,
                "experience_level": user_profile.experience_level if user_profile else "unknown"
            },
            "filters_applied": {
                "min_match_score": min_match_score,
                "item_type": item_type,
                "sort_by": sort_by
            },
            "performance": result.get('performance_stats', {}),
            "cache_status": "refreshed" if result.get('cache_refreshed') else "cached"
        }

        return JSONResponse(content={"success": True, "data": response_data})

    except Exception as err:
        print(f"Enhanced recommendations error: {err}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:
        session.close()


async def generate_recommendation_insights(user_id: int, recommendations: List[Dict],
                                           item_type: str) -> Dict[str, Any]:
    """Generate insights about recommendations"""
    try:
        if not recommendations:
            return {}

        # Calculate insights
        avg_match_score = sum(rec.get('match_score', 0)
                              for rec in recommendations) / len(recommendations)

        # Most common missing skills
        all_missing_skills = []
        for rec in recommendations:
            all_missing_skills.extend(rec.get('missing_skills', []))

        from collections import Counter
        skill_gaps = Counter(all_missing_skills).most_common(5)

        # Industry/company analysis
        if item_type in ['job', 'external_job']:
            companies = [rec.get('item_data', {}).get('company', '')
                         for rec in recommendations]
            top_companies = Counter(companies).most_common(3)

            locations = [rec.get('item_data', {}).get('location', '')
                         for rec in recommendations]
            top_locations = Counter(locations).most_common(3)
        else:
            top_companies = []
            top_locations = []

        return {
            "average_match_score": round(avg_match_score, 1),
            "total_opportunities": len(recommendations),
            "skill_gaps": [{"skill": skill, "frequency": count} for skill, count in skill_gaps],
            "top_companies": [{"company": comp, "opportunities": count} for comp, count in top_companies],
            "top_locations": [{"location": loc, "opportunities": count} for loc, count in top_locations],
            "readiness_level": (
                "Excellent - You're ready for most opportunities" if avg_match_score >= 70 else
                "Good - Consider skill development for better matches" if avg_match_score >= 50 else
                "Developing - Focus on building key skills"
            )
        }

    except Exception as e:
        print(f"Error generating insights: {e}")
        return {}


@router.get("/detailed_match_analysis/{item_type}/{item_id}/")
async def get_detailed_match_analysis_v2(
    item_type: str,
    item_id: int,
    request: Request,
    force_refresh: bool = Query(
        False, description="Force cache refresh"),
    include_improvement_plan: bool = Query(
        True, description="Include improvement plan"),
    include_similar_items: bool = Query(
        True, description="Include similar opportunities")
):
    """
    Enhanced detailed match analysis with improvement suggestions
    """

    user_id = request.state.user["id"]

    try:
        controller: EnhancedMatchingController = get_enhanced_controller(
            session)
        matching_service = JobCourseMatchingService(
            session, controller.llm_processor)
        user_profile = matching_service.get_user_profile_from_db(user_id)

        if not user_profile:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "User profile not found"}
            )

        # Get item data
        item_data = None
        if item_type == "job":
            job = session.query(Job).filter(Job.id == item_id).first()
            if job:
                item_data = {
                    "id": job.id, "title": job.title, "company": job.company,
                    "description": job.description, "skills": job.skills or [],
                    "requirements": job.requirements or [], "salary": job.salary,
                    "location": job.location, "type": job.type,
                    "mode": "job"
                }
        elif item_type == "external_job":
            job = session.query(ExternalJob).filter(
                ExternalJob.id == item_id).first()
            if job:
                item_data = {
                    "id": job.id, "title": job.title, "company": job.company,
                    "description": job.description or "", "skills": job.skills or [],
                    "salary_min": job.salary_min, "salary_max": job.salary_max,
                    "location": job.location, "job_type": job.job_type,
                    "apply_url": job.apply_url, "source": job.source,
                    "mode": "external_job"
                }
        elif item_type == "course":
            course = session.query(Course).filter(Course.id == item_id).first()
            if course:
                item_data = {
                    "id": course.id, "title": course.title, "description": course.description,
                    "skills": course.skills or [], "requirements": course.requirements or [],
                    "language": course.language, "lessons": course.lessons or []
                }

        if not item_data:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Item not found"}
            )

        # Calculate comprehensive match
        match_result = controller.llm_processor.matching_engine.calculate_comprehensive_match(
            user_profile, item_data, item_type
        )

        # Generate improvement plan
        improvement_plan = {}
        if include_improvement_plan and match_result.missing_skills:
            improvement_plan = await generate_improvement_plan(
                user_profile, match_result.missing_skills, item_data, item_type
            )

        # Find similar items
        similar_items = []
        if include_similar_items:
            similar_items = await find_similar_opportunities(
                item_data, item_type, user_profile, controller, limit=3, force_refresh=force_refresh
            )

        analysis = {
            "item": item_data,
            "match_analysis": {
                "overall_score": match_result.match_score,
                "skill_compatibility": round((match_result.skill_match_count / max(match_result.total_required_skills, 1)) * 100, 1),
                "matched_skills": match_result.matched_skills,
                "missing_skills": match_result.missing_skills,
                "skill_gap_count": len(match_result.missing_skills),
                "readiness_assessment": get_readiness_assessment(match_result.match_score)
            },
            "improvement_plan": improvement_plan,
            "similar_opportunities": similar_items,
            "user_profile_summary": {
                "total_skills": len(user_profile.skills),
                "experience_level": user_profile.experience_level,
                "profile_strength": calculate_profile_strength(user_profile)
            }
        }

        return JSONResponse(content={"success": True, "data": analysis})

    except Exception as err:
        print(f"Detailed analysis error: {err}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:
        session.close()


async def generate_improvement_plan(user_profile: UserProfile, missing_skills: List[str],
                                    item_data: Dict, item_type: str) -> Dict[str, Any]:
    """Generate personalized improvement plan"""
    try:

        plan_prompt = f"""
            Create a personalized improvement plan for this user to increase their match score for this {item_type}.

            User's Current Skills: {', '.join(user_profile.skills) if user_profile.skills else 'None'}
            Missing Skills Needed: {', '.join(missing_skills)}
            Target {item_type.title()}: {item_data.get('title', 'Unknown')}

            Provide a structured improvement plan in JSON format with the following exact structure:
            ```json
            {{
                "improvement_steps": [
                    {{
                        "step_number": 1,
                        "title": "Step Title",
                        "description": "Detailed description of the step",
                        "estimated_time": "Duration (e.g., 2 weeks)",
                        "resources": ["Resource 1", "Resource 2"],
                        "skills_to_acquire": ["Skill 1", "Skill 2"]
                    }},
                    ...
                ],
                "estimated_time": "Total duration (e.g., 3 months)",
                "resources": ["General Resource 1", "General Resource 2"]
            }}"""

        # Use custom_prompt instead of prompt_template
        plan_response = llm.query_llm_with_template(
            "", custom_prompt=plan_prompt)

        return llm.parse_json_output(plan_response)
    except Exception as e:
        print(f"Error generating improvement plan: {e}")
        return {}


async def find_similar_opportunities(item_data: Dict, item_type: str, user_profile: UserProfile,
                                     controller: EnhancedMatchingController, limit: int = 3, force_refresh: bool = False) -> List[Dict]:
    """Find similar opportunities that might be good matches"""
    try:
        # Get recommendations of the same type
        result = controller.process_user_matching_request(
            user_profile.user_id, item_type, force_refresh=force_refresh, limit=20
        )

        recommendations = result.get('recommendations', [])

        # Filter out the current item and find similar ones
        similar = []
        current_skills = set(item_data.get('skills', []))
        current_title_words = set(item_data.get('title', '').lower().split())

        for rec in recommendations:
            if rec.get('item_id') == item_data.get('id'):
                continue  # Skip current item

            rec_data = rec.get('item_data', {})
            rec_skills = set(rec_data.get('skills', []))
            rec_title_words = set(rec_data.get('title', '').lower().split())

            # Calculate similarity based on skills and title
            skill_similarity = len(current_skills.intersection(
                rec_skills)) / max(len(current_skills.union(rec_skills)), 1)
            title_similarity = len(current_title_words.intersection(
                rec_title_words)) / max(len(current_title_words.union(rec_title_words)), 1)

            combined_similarity = (skill_similarity * 0.7) + \
                (title_similarity * 0.3)

            if combined_similarity > 0.3:  # Minimum similarity threshold
                similar.append({
                    **rec,
                    "similarity_score": round(combined_similarity * 100, 1)
                })

        # Sort by similarity and match score
        similar.sort(key=lambda x: (x.get('similarity_score', 0) +
                     x.get('match_score', 0)), reverse=True)

        return similar[:limit]

    except Exception as e:
        print(f"Error finding similar opportunities: {e}")
        return []


def get_readiness_assessment(match_score: float) -> str:
    """Get readiness assessment based on match score"""
    if match_score >= 80:
        return "Excellent match - You're well-qualified for this opportunity"
    elif match_score >= 60:
        return "Good match - You meet most requirements with minor gaps"
    elif match_score >= 40:
        return "Fair match - Some skill development recommended"
    else:
        return "Limited match - Significant preparation needed"


def calculate_profile_strength(user_profile: UserProfile) -> str:
    """Calculate overall profile strength"""
    score = 0

    # Skills count
    if len(user_profile.skills) >= 10:
        score += 30
    elif len(user_profile.skills) >= 5:
        score += 20
    else:
        score += 10

    # Resume quality (length as proxy)
    if user_profile.resume_text and len(user_profile.resume_text) > 2000:
        score += 25
    elif user_profile.resume_text and len(user_profile.resume_text) > 1000:
        score += 15
    else:
        score += 5

    # Experience level
    if user_profile.experience_level in ['Senior Level', 'Executive']:
        score += 25
    elif user_profile.experience_level == 'Mid Level':
        score += 20
    else:
        score += 10

    # Remaining 20 points for completeness
    score += 20

    if score >= 80:
        return "Strong"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Developing"
    else:
        return "Basic"


@router.api_route("/preferences", methods=["GET", "PATCH", "PUT"])
async def user_matching_preferences(request: Request):
    """Get or update user's matching preferences"""
    user_id = request.state.user["id"]

    try:
        controller = get_enhanced_controller(session)

        if request.method == "GET":
            preferences = controller.optimized_service.get_user_preferences(
                user_id)
            return JSONResponse(content={"success": True, "data": preferences})

        elif request.method in ["PATCH", "PUT"]:
            preferences = await request.json()
            controller.optimized_service.update_user_preferences(
                user_id, preferences)
            return JSONResponse(content={
                "success": True,
                "message": "Preferences updated successfully. Cache cleared for fresh recommendations."
            })

    except Exception as err:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:
        session.close()


@router.get("/extract_skills_from_job_recommendations")
async def extract_skills_from_job_recommendations(request: Request, min_match_score: int = 40, limit: int = 20):

    try:
        user_id = request.state.user["id"]

        job_service = JobCourseMatchingService(session)

        job_recommendations = job_service.get_recommended_jobs(
            user_id=user_id, min_match_score=min_match_score, limit=limit)

        skills_data = job_service.extract_skills_from_job_recommendations(
            job_recommendations)

        return sendSuccess(skills_data)
    except Exception as err:
        session.rollback()
        print(err)
        return sendError(
            "unable to fetch skills", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        session.close()


@router.post("/refresh_all_matches")
async def refresh_all_user_matches(request: Request, background_tasks: BackgroundTasks):
    user_id = request.state.user["id"]

    try:
        controller = get_enhanced_controller(session)

        # Clear all existing cache
        controller.cache_manager.invalidate_user_cache(user_id)

        # Schedule comprehensive background refresh
        background_tasks.add_task(
            comprehensive_match_refresh, user_id, force_refresh=True)

        return JSONResponse(content={
            "success": True,
            "message": "Comprehensive match refresh initiated. This may take a few minutes.",
            "estimated_completion": "2-5 minutes"
        })

    except Exception as err:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:

        session.close()


async def comprehensive_match_refresh(user_id: int, force_refresh: bool = False, limit: int = 50):
    """Comprehensive background refresh of all matches"""
    try:
        controller = get_enhanced_controller(session)

        print(f"Starting comprehensive refresh for user {user_id}")

        # Refresh internal jobs
        job_result = controller.process_user_matching_request(
            user_id, 'job', force_refresh=force_refresh, limit=limit
        )

        # Refresh external jobs (now properly supported)
        external_result = controller.process_user_matching_request(
            user_id, 'external_job', force_refresh=force_refresh, limit=limit
        )

        # # Refresh courses
        course_result = controller.process_user_matching_request(
            user_id, 'course', force_refresh=force_refresh, limit=limit
        )

        print(f"Comprehensive refresh completed for user {user_id}:")
        print(
            f"  - Internal Jobs: {len(job_result.get('recommendations', []))}")
        print(
            f"  - External Jobs: {len(external_result.get('recommendations', []))}")
        print(f"  - Courses: {len(course_result.get('recommendations', []))}")

        # Check for errors
        if job_result.get('error'):
            print(f"  - Internal Jobs Error: {job_result['error']}")
        if external_result.get('error'):
            print(f"  - External Jobs Error: {external_result['error']}")
        if course_result.get('error'):
            print(f"  - Courses Error: {course_result['error']}")

        # Cleanup and optimize
        cleanup_result = controller.cleanup_and_optimize(user_id)

        # await ws_manager.send_user_notification(str(user_id),
        #                                         {"type": "jobs_updated",
        #                                          "message": "Upload complete!"}
        #                                         )

        print(
            f"  - Cache cleanup: {cleanup_result.get('expired_entries_removed', 0)} entries removed")

        return {
            "job_recommendations": job_result.get('recommendations', []),
            "external_job_recommendations": external_result.get('recommendations', []),
            "course_recommendations": course_result.get('recommendations', []),
            "cleanup_stats": cleanup_result
        }

    except Exception as e:
        print(f"Comprehensive refresh error for user {user_id}: {e}")
        return {"error": str(e)}
    finally:

        session.close()


@router.get("/system_stats")
async def get_system_statistics(request: Request):
    """Get system performance and cache statistics (admin only)"""
    # Add admin check here based on your auth system
    user_role = request.state.user.get("role")

    if user_role.lower() != "admin":
        return JSONResponse(status_code=403, content={"success": False, "error": "Admin access required"})

    try:
        controller = get_enhanced_controller(session)

        # Get cache statistics
        cache_stats = controller.cache_manager.get_cache_stats()

        # Get performance metrics
        performance_stats = controller.performance_monitor.get_stats()

        # Get database statistics
        total_users = session.query(User).count()
        total_jobs = session.query(Job).count()
        total_external_jobs = session.query(ExternalJob).filter(
            ExternalJob.is_active == True).count()
        total_courses = session.query(Course).filter(
            Course.isActive == True).count()

        # Recent activity
        recent_uploads = session.query(UserResume).filter(
            UserResume.created >= datetime.datetime.utcnow() - datetime.timedelta(days=7)
        ).count()

        stats = {
            "cache_statistics": cache_stats,
            "performance_metrics": performance_stats,
            "database_counts": {
                "total_users": total_users,
                "total_jobs": total_jobs,
                "total_external_jobs": total_external_jobs,
                "total_courses": total_courses
            },
            "recent_activity": {
                "resume_uploads_last_7_days": recent_uploads
            },
            "system_health": "healthy",  # Add actual health checks
            "last_updated": datetime.datetime.utcnow().isoformat()
        }

        return JSONResponse(content={"success": True, "data": stats})

    except Exception as err:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:
        session.close()


@router.post("/cleanup_system")
async def cleanup_system_cache(request: Request, background_tasks: BackgroundTasks):
    """Cleanup expired cache and optimize system (admin only)"""
    user_role = request.state.user.get("role")

    if user_role.lower() != "admin":
        return JSONResponse(status_code=403, content={"success": False, "error": "Admin access required"})

    try:
        controller = get_enhanced_controller(session)

        # Schedule cleanup in background
        background_tasks.add_task(perform_system_cleanup, controller)

        return JSONResponse(content={
            "success": True,
            "message": "System cleanup initiated"
        })

    except Exception as err:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(err)}
        )
    finally:
        session.close()


async def perform_system_cleanup(controller: EnhancedMatchingController):
    """Perform system cleanup tasks"""
    try:
        cleanup_result = controller.cleanup_and_optimize()
        print(f"System cleanup completed: {cleanup_result}")

    except Exception as e:
        print(f"System cleanup error: {e}")


# OLD APIS

@router.post("/experience", status_code=status.HTTP_201_CREATED)
async def add_experience(request: Request, data: ExperienceData):
    exp = Experience()
    try:
        exp.user_id = request.state.user["seeker_id"]
        exp.company_name = data.company_name
        exp.job_title = data.job_title
        exp.start_date = data.start_date
        exp.end_date = data.end_date
        exp.is_remote = data.is_remote
        exp.has_completed = data.has_completed
        exp.tasks = data.tasks
        session.add(exp)
        session.commit()

        # new_exp = session.query(Experience).filter(Experience.id == exp.id).first()
        return sendSuccess("created")
    except Exception as err:
        print(err)
        session.rollback()
        sendError("unable to add experience")
    finally:
        session.close()


@router.delete("/experience/{id}", status_code=200)
async def delete_experience(id: str, request: Request):
    try:
        user_id = request.state.user["id"]
        exp = (
            session.query(Experience)
            .filter(Experience.id == id, Experience.user_id == user_id)
            .first()
        )
        session.delete(exp)
        session.commit()
        return sendSuccess("deleted")
    except Exception as err:
        session.rollback()
        return sendError("unable to delete experience")
    finally:
        session.close()


@router.put("/experience", status_code=200)
async def update_experience(request: Request, data: ExperienceData):
    try:
        user_id = request.state.user["id"]
        exp = (
            session.query(Experience)
            .filter(Experience.id == data.id, Experience.user_id == user_id)
            .first()
        )
        if exp is None:
            return sendError("experience not found")

        exp.company_name = data.company_name
        exp.job_title = data.job_title
        exp.start_date = data.start_date
        exp.end_date = data.end_date
        exp.is_remote = data.is_remote
        exp.has_completed = data.has_completed
        exp.tasks = data.tasks
        session.commit()
        return sendSuccess("saved")
    except Exception as err:
        session.rollback()
        print(err)
        sendError("unable to update experience")
    finally:
        session.close()


class EducationData(BaseModel):
    id: Optional[int] = None
    program: str
    institution: str
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    has_completed: bool = False


@router.put("/education")
def update_education(request: Request, data: EducationData):
    try:
        user_id = request.state.user["id"]
        edu = (
            session.query(Education)
            .filter(Education.id == data.id, Education.user_id == user_id)
            .first()
        )
        if edu is None:
            return sendError("not found")
        edu.program = data.program
        edu.institution = data.institution
        edu.start_date = data.start_date
        edu.end_date = data.end_date
        edu.has_completed = data.has_completed
        session.commit()
        return sendSuccess("updated")

    except Exception as err:
        session.rollback()
        return sendError("uanle to update education")
    finally:
        session.close()


@router.delete("/education/{id}")
def delete_education(request: Request, id: str):
    try:
        user_id = request.state.user["id"]
        edu = (
            session.query(Education)
            .filter(Education.id == id, Education.user_id == user_id)
            .first()
        )
        if edu is None:
            return sendError("not found")
        session.delete(edu)
        session.commit()

        return sendSuccess("deleted")
    except Exception as err:
        session.rollback()
        return sendError("uanle to update education")
    finally:
        session.close()


@router.post("/education")
def add_education(request: Request, data: EducationData):
    ed = Education()

    ed.user_id = request.state.user["id"]
    ed.program = data.program
    ed.institution = data.institution
    ed.start_date = data.start_date
    ed.end_date = data.end_date
    ed.has_completed = data.has_completed

    try:
        session.add(ed)
        session.commit()
        return sendSuccess(ed)
    except Exception as err:
        session.rollback()
        return sendError("unable to add education")
    finally:
        session.close()


@router.get("/skill")
async def get_skills(request: Request):
    try:
        res = (
            session.query(Skill)
            .join(JobSeekerSkill)
            .filter(JobSeekerSkill.user_id == request.state.user["id"])
            .all()
        )
        return sendSuccess(res)
    except Exception as err:
        session.rollback()
        print(err)
        return sendError(
            "unable to fetch skills", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        session.close()


@router.get("/skills")
async def get_all_skills(request: Request):
    q = request.query_params.get("q")
    # serrch
    try:
        query = session.query(Skill)
        if q is not None:
            query = query.filter(Skill.lower.ilike("%{}%".format(q.lower())))
        res = query.limit(50).all()

        return sendSuccess(list(map(lambda x: {"id": x.id, "name": x.name}, res)))
    except Exception as err:
        session.rollback()
        print(err)
        return sendError(
            "unable to fetch skills", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        session.close()


@router.get("/recommend_skills")
async def recommend_skills(request: Request, min_match_score: int = 40, limit: int = 20):

    try:
        user_id = request.state.user["id"]

        job_service = JobCourseMatchingService(session)

        job_recommendations = job_service.get_recommended_jobs(
            user_id=user_id, min_match_score=min_match_score, limit=limit)

        skills_data = job_service.extract_skills_from_job_recommendations(
            job_recommendations)

        # Convert skills to single list with name and type
        all_skills = []

        # Add missing skills
        for skill in skills_data.get("missing_skills", []):
            all_skills.append({"name": skill, "type": "missing_skill"})

        # Add user skills
        for skill in skills_data.get("user_skills", []):
            all_skills.append({"name": skill, "type": "user_skill"})

        return sendSuccess(all_skills)
    except Exception as err:
        session.rollback()
        print(err)
        return sendError(
            "unable to fetch skills", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        session.close()

# skill model for request body


class SkillData(BaseModel):
    skills: List[int]


@router.post("/skill")
async def add_skill(request: Request, data: SkillData):
    user_id = request.state.user["id"]
    try:
        exist = (
            session.query(JobSeekerSkill)
            .filter(JobSeekerSkill.user_id == user_id)
            .all()
        )
        for i in exist:
            session.delete(i)
        for skill_id in data.skills:
            count = session.query(Skill).filter(Skill.id == skill_id).count()
            if count > 0:
                print("adding ", skill_id, " to ", user_id)
                sk = JobSeekerSkill()
                sk.skill_id = skill_id
                sk.user_id = user_id
                session.add(sk)

        session.commit()
        return sendSuccess("skills uploaded")
    except Exception as err:
        session.rollback()
        return sendError(err.args)
    finally:
        session.close()


# TODO: remove
class SkillCreate(BaseModel):
    skills: List[str]


@router.post("/create_skill", status_code=status.HTTP_201_CREATED)
async def create_skill(data: SkillCreate):
    try:
        for name in data.skills:
            existing_skill = session.query(
                Skill).filter(Skill.name == name).count()
            if existing_skill < 1:
                sk = Skill(name)
                session.add(sk)
                session.commit()

        return sendSuccess("created")
    except Exception as err:
        session.rollback()
        return sendError(err.args)
    finally:
        session.close()


class UpdateProfile(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    about: Optional[str] = None
    location: Optional[str] = None
    portfolio: Optional[str] = None
    languages: Optional[List[str]] = None


@router.put("/profile")
async def update_profile(request: Request, data: UpdateProfile):
    # print(data)
    try:
        user = session.query(User).filter(
            User.id == request.state.user["id"]).first()
        profile = (
            session.query(JobSeeker).filter(
                JobSeeker.id == user.profile_id).first()
        )
        if data.name is not None:
            user.name = data.name
        if data.about is not None:
            profile.about = data.about
        if data.location is not None:
            profile.location = data.location
        if data.portfolio is not None:
            profile.portfolio = data.portfolio
        if data.languages is not None:
            profile.languages = data.languages

        session.commit()
        return sendSuccess("Profile updated successfully")

    except Exception as err:
        session.rollback()
        print(err)
        return sendError("internal server error", 500)
    finally:
        session.close()


@router.post("/image", status_code=status.HTTP_201_CREATED)
async def upload_image(img: UploadFile, request: Request):
    user_id = request.state.user["id"]

    try:
        user = session.query(User).filter(User.id == user_id).first()

        new_file = await img.read()
        fileSha = getSha(new_file)
        ex_chunk = img.filename.split(".")
        ext = ex_chunk[len(ex_chunk) - 1:][0]
        filename = str(uuid.uuid4()) + "." + ext
        print(filename, ext, sep=" |  ")
        dbImg = File(
            data=new_file, filename=filename, type=img.content_type, sha=fileSha
        )
        session.add(dbImg)
        user.profile_image = filename

        session.commit()
        return sendSuccess("uploaded")
    except Exception as err:
        session.rollback()
        return sendError(err.args)
    finally:
        session.close()


class DeleteResume(BaseModel):
    url: str


@router.delete("/resume")
async def remove_resume(request: Request, data: DeleteResume):
    try:
        user_id = request.state.user["id"]
        url = data.url
        chunk = url.split("/")
        filename = chunk[len(chunk) - 1:][0]
        print("filename = ", filename)
        resume = (
            session.query(UserResume)
            .filter(UserResume.user_id == user_id, UserResume.filename == filename)
            .first()
        )
        resume_file = session.query(File).filter(
            File.filename == filename).first()
        session.delete(resume)
        session.delete(resume_file)
        session.commit()
        return sendSuccess("deleted")
    except Exception as err:
        session.rollback()
        print(err)
        return sendError("unable to remove resume")
    finally:
        session.close()


@app_router.get("/file/{filename}")
async def stream_file(filename: str, request: Request, resp: Response):
    try:
        file = session.query(File).filter(File.filename == filename).first()
        if file == None:
            return sendError("file not found")
        return Response(content=file.data, media_type=file.type)
    except Exception as err:
        session.rollback()
        print(err)
        return sendError("unable to get file", 500)
    finally:
        session.close()


# Job Recommendation System
def calculate_job_match_score(user_skills, job_skills, job_location, user_location=None):
    """Calculate match score between user and job based on skills and location"""
    if not user_skills or not job_skills:
        return 0.0, 0, job_skills or []

    # Convert to lowercase for comparison
    user_skills_lower = [skill.lower() for skill in user_skills]
    job_skills_lower = [skill.lower() for skill in job_skills]

    # Calculate skill overlap
    matching_skills = set(user_skills_lower) & set(job_skills_lower)
    skill_match_count = len(matching_skills)
    skill_match_percentage = skill_match_count / \
        len(job_skills_lower) if job_skills_lower else 0

    # Find missing skills
    missing_skills = [
        skill for skill in job_skills if skill.lower() not in user_skills_lower]

    # Base score from skill matching (70% weight)
    skill_score = skill_match_percentage * 70

    # Location matching (20% weight)
    location_score = 0
    if user_location and job_location:
        if user_location.lower() in job_location.lower() or job_location.lower() in user_location.lower():
            location_score = 20

    # Experience bonus (10% weight) - simplified for now
    experience_score = 10 if skill_match_count > 0 else 0

    total_score = skill_score + location_score + experience_score
    return min(total_score, 100.0), skill_match_count, missing_skills


@router.get("/recommend_jobs")
async def recommend_jobs(request: Request):
    """Get personalized job recommendations for the authenticated user"""
    limit_q = request.query_params.get("limit")
    limit = 20
    if limit_q is not None:
        limit = int(limit_q)

    user_id = request.state.user["id"]

    try:
        # Get user's skills
        user_skills = (
            session.query(JobSeekerSkill)
            .join(JobSeekerSkill.skill)
            .filter(JobSeekerSkill.user_id == user_id)
            .all()
        )
        user_skill_names = [skill.skill.name for skill in user_skills]

        # Get user's location from profile
        user = session.query(User).join(
            User.profile).filter(User.id == user_id).first()
        user_location = user.profile.location if user and user.profile else None

        # Get all active jobs
        jobs = session.query(Job).all()

        job_recommendations = []

        for job in jobs:
            # Calculate match score
            match_score, skill_match_count, missing_skills = calculate_job_match_score(
                user_skill_names, job.skills, job.location, user_location
            )

            if match_score > 0:  # Only include jobs with some match
                job_data = {
                    "id": job.id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "salary": job.salary,
                    "type": job.type,
                    "description": job.description,
                    "skills": job.skills,
                    "requirements": job.requirements,
                    "expiry": job.expiry.isoformat() if job.expiry else None,
                    "match_score": round(match_score, 2),
                    "skill_match_count": skill_match_count,
                    "missing_skills": missing_skills,
                    "image": job.image
                }
                job_recommendations.append(job_data)

        # Sort by match score (highest first) and limit results
        job_recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        job_recommendations = job_recommendations[:limit]

        # Store job matches in database for analytics
        for job_rec in job_recommendations:
            existing_match = session.query(JobMatch).filter(
                JobMatch.user_id == user_id,
                JobMatch.job_id == job_rec["id"]
            ).first()

            if existing_match:
                # Update existing match
                existing_match.match_score = job_rec["match_score"]
                existing_match.skill_match_count = job_rec["skill_match_count"]
                existing_match.missing_skills = job_rec["missing_skills"]
            else:
                # Create new match record
                job_match = JobMatch(
                    user_id=user_id,
                    job_id=job_rec["id"],
                    match_score=job_rec["match_score"],
                    skill_match_count=job_rec["skill_match_count"],
                    missing_skills=job_rec["missing_skills"]
                )
                session.add(job_match)

        session.commit()
        return sendSuccess(job_recommendations)

    except Exception as err:
        session.rollback()
        print(f"Error in recommend_jobs: {err}")
        return sendError(str(err))
    finally:
        session.close()


@router.get("/job_preferences")
async def get_job_preferences(request: Request):
    """Get user's job preferences"""
    user_id = request.state.user["id"]

    try:
        preferences = session.query(UserJobPreferences).filter(
            UserJobPreferences.user_id == user_id
        ).first()

        if preferences:
            pref_data = {
                "preferred_locations": preferences.preferred_locations,
                "preferred_job_types": preferences.preferred_job_types,
                "min_salary": preferences.min_salary,
                "max_salary": preferences.max_salary,
                "remote_ok": preferences.remote_ok
            }
            return sendSuccess(pref_data)
        else:
            return sendSuccess({
                "preferred_locations": [],
                "preferred_job_types": [],
                "min_salary": None,
                "max_salary": None,
                "remote_ok": False
            })

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


class JobPreferencesData(BaseModel):
    preferred_locations: Optional[List[str]] = []
    preferred_job_types: Optional[List[str]] = []
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    remote_ok: Optional[bool] = False


@router.post("/job_preferences")
async def update_job_preferences(request: Request, data: JobPreferencesData):
    """Update user's job preferences"""
    user_id = request.state.user["id"]

    try:
        existing_prefs = session.query(UserJobPreferences).filter(
            UserJobPreferences.user_id == user_id
        ).first()

        if existing_prefs:
            # Update existing preferences
            existing_prefs.preferred_locations = data.preferred_locations
            existing_prefs.preferred_job_types = data.preferred_job_types
            existing_prefs.min_salary = data.min_salary
            existing_prefs.max_salary = data.max_salary
            existing_prefs.remote_ok = data.remote_ok
        else:
            # Create new preferences
            new_prefs = UserJobPreferences(
                user_id=user_id,
                preferred_locations=data.preferred_locations,
                preferred_job_types=data.preferred_job_types,
                min_salary=data.min_salary,
                max_salary=data.max_salary,
                remote_ok=data.remote_ok
            )
            session.add(new_prefs)

        session.commit()
        return sendSuccess("Job preferences updated successfully")

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


# External Job Management
@router.post("/scrape_jobs")
async def scrape_external_jobs(request: Request):
    """Scrape jobs from external sources and store in database"""
    try:
        from services.job_scraper import JobScraper

        scraper = JobScraper()
        scraped_jobs = scraper.scrape_all_sources(limit_per_source=10)

        saved_count = 0
        for job_data in scraped_jobs:
            # Check if job already exists
            existing_job = session.query(ExternalJob).filter(
                ExternalJob.title == job_data['title'],
                ExternalJob.company == job_data['company'],
                ExternalJob.source == job_data['source']
            ).first()

            if not existing_job:
                external_job = ExternalJob(
                    title=job_data['title'],
                    company=job_data['company'],
                    location=job_data['location'],
                    description=job_data['description'],
                    skills=job_data['skills'],
                    salary_min=job_data['salary_min'],
                    salary_max=job_data['salary_max'],
                    job_type=job_data['job_type'],
                    apply_url=job_data['apply_url'],
                    source=job_data['source'],
                    posted_date=job_data['posted_date']
                )
                session.add(external_job)
                saved_count += 1

        session.commit()
        return sendSuccess(f"Scraped and saved {saved_count} new jobs from {len(scraped_jobs)} total")

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


@router.get("/external_jobs")
async def get_external_jobs(request: Request):
    """Get all external jobs"""
    limit_q = request.query_params.get("limit")
    source_q = request.query_params.get("source")

    limit = 50
    if limit_q is not None:
        limit = int(limit_q)

    try:
        query = session.query(ExternalJob).filter(
            ExternalJob.is_active.is_(True))

        if source_q:
            query = query.filter(ExternalJob.source == source_q)

        external_jobs = query.order_by(
            ExternalJob.scraped_date.desc()).limit(limit).all()

        job_list = []
        for job in external_jobs:
            job_dict = {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description,
                "skills": job.skills,
                "salary_min": job.salary_min,
                "salary_max": job.salary_max,
                "job_type": job.job_type,
                "apply_url": job.apply_url,
                "source": job.source,
                "posted_date": job.posted_date.isoformat() if job.posted_date else None,
                "scraped_date": job.scraped_date.isoformat(),
                "is_enabled": job.is_enabled,
                "is_active": job.is_active,

            }
            job_list.append(job_dict)

        return sendSuccess(job_list)

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


@router.get("/recommend_external_jobs")
async def recommend_external_jobs(request: Request):
    """Get personalized external job recommendations"""
    limit_q = request.query_params.get("limit")
    limit = 20
    if limit_q is not None:
        limit = int(limit_q)

    user_id = request.state.user["id"]

    try:
        # Get user's skills
        user_skills = (
            session.query(JobSeekerSkill)
            .join(JobSeekerSkill.skill)
            .filter(JobSeekerSkill.user_id == user_id)
            .all()
        )
        user_skill_names = [skill.skill.name for skill in user_skills]

        # Get user's location from profile
        user = session.query(User).join(
            User.profile).filter(User.id == user_id).first()
        user_location = user.profile.location if user and user.profile else None

        # Get all active and enabled external jobs
        external_jobs = session.query(ExternalJob).filter(
            ExternalJob.is_active.is_(True),
            ExternalJob.is_enabled.is_(True)
        ).all()

        job_recommendations = []

        for job in external_jobs:
            # Calculate match score using the same algorithm
            match_score, skill_match_count, missing_skills = calculate_job_match_score(
                user_skill_names, job.skills, job.location, user_location
            )

            if match_score > 0:  # Only include jobs with some match
                job_data = {
                    "id": job.id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description,
                    "skills": job.skills,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "job_type": job.job_type,
                    "apply_url": job.apply_url,
                    "source": job.source,
                    "posted_date": job.posted_date.isoformat() if job.posted_date else None,
                    "match_score": round(match_score, 2),
                    "skill_match_count": skill_match_count,
                    "missing_skills": missing_skills,
                    "is_external": True
                }
                job_recommendations.append(job_data)

        # Sort by match score and limit results
        job_recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        job_recommendations = job_recommendations[:limit]

        # Store external job matches for analytics
        for job_rec in job_recommendations:
            existing_match = session.query(ExternalJobMatch).filter(
                ExternalJobMatch.user_id == user_id,
                ExternalJobMatch.external_job_id == job_rec["id"]
            ).first()

            if existing_match:
                existing_match.match_score = job_rec["match_score"]
                existing_match.skill_match_count = job_rec["skill_match_count"]
                existing_match.missing_skills = job_rec["missing_skills"]
            else:
                external_match = ExternalJobMatch(
                    user_id=user_id,
                    external_job_id=job_rec["id"],
                    match_score=job_rec["match_score"],
                    skill_match_count=job_rec["skill_match_count"],
                    missing_skills=job_rec["missing_skills"]
                )
                session.add(external_match)

        session.commit()
        return sendSuccess(job_recommendations)

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


@router.get("/recommend_all_jobs")
async def recommend_all_jobs(request: Request):
    """Get combined recommendations from both internal and external jobs"""
    limit_q = request.query_params.get("limit")
    limit = 20
    if limit_q is not None:
        limit = int(limit_q)

    try:
        # Get internal job recommendations
        internal_resp = await recommend_jobs(request)
        internal_jobs = internal_resp.body.decode(
        ) if hasattr(internal_resp, 'body') else []

        # Get external job recommendations
        external_resp = await recommend_external_jobs(request)
        external_jobs = external_resp.body.decode(
        ) if hasattr(external_resp, 'body') else []

        # For simplicity, let's call the functions directly
        user_id = request.state.user["id"]

        # Get user skills
        user_skills = (
            session.query(JobSeekerSkill)
            .join(JobSeekerSkill.skill)
            .filter(JobSeekerSkill.user_id == user_id)
            .all()
        )
        user_skill_names = [skill.skill.name for skill in user_skills]

        # Get user location
        user = session.query(User).join(
            User.profile).filter(User.id == user_id).first()
        user_location = user.profile.location if user and user.profile else None

        all_recommendations = []

        # Add internal jobs
        internal_jobs = session.query(Job).all()
        for job in internal_jobs:
            match_score, skill_match_count, missing_skills = calculate_job_match_score(
                user_skill_names, job.skills, job.location, user_location
            )

            if match_score > 0:
                job_data = {
                    "id": job.id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description,
                    "skills": job.skills,
                    "salary": job.salary,
                    "job_type": job.type,
                    "apply_url": None,  # Internal jobs don't have external apply URLs
                    "source": "Internal",
                    "match_score": round(match_score, 2),
                    "skill_match_count": skill_match_count,
                    "missing_skills": missing_skills,
                    "is_external": False
                }
                all_recommendations.append(job_data)

        # Add external jobs (only enabled ones for mobile users)
        external_jobs = session.query(ExternalJob).filter(
            ExternalJob.is_active.is_(True),
            ExternalJob.is_enabled.is_(True)
        ).all()
        for job in external_jobs:
            match_score, skill_match_count, missing_skills = calculate_job_match_score(
                user_skill_names, job.skills, job.location, user_location
            )

            # Include all external jobs, even with 0 match score
            job_data = {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description,
                "skills": job.skills,
                "salary_min": job.salary_min,
                "salary_max": job.salary_max,
                "job_type": job.job_type,
                "apply_url": job.apply_url,
                "source": job.source,
                "match_score": round(match_score, 2),
                "skill_match_count": skill_match_count,
                "missing_skills": missing_skills,
                "is_external": True
            }
            all_recommendations.append(job_data)

        # Sort by match score and limit
        all_recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        all_recommendations = all_recommendations[:limit]

        return sendSuccess(all_recommendations)

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


# Admin functions for external job management
@router.post("/external_jobs/{job_id}/enable")
async def enable_external_job(job_id: int, request: Request):
    """Enable an external job (admin only)"""
    user_id = request.state.user["id"]

    try:
        # Check if user is admin
        user = session.query(User).filter(User.id == user_id).first()
        if not user or user.role != "ADMIN":
            return sendError("Unauthorized: Admin access required", 403)

        external_job = session.query(ExternalJob).filter(
            ExternalJob.id == job_id).first()
        if not external_job:
            return sendError("External job not found", 404)

        external_job.is_enabled = True
        session.commit()

        return sendSuccess("External job enabled successfully")

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


@router.post("/external_jobs/{job_id}/disable")
async def disable_external_job(job_id: int, request: Request):
    """Disable an external job (admin only)"""
    user_id = request.state.user["id"]

    try:
        # Check if user is admin
        user = session.query(User).filter(User.id == user_id).first()
        if not user or user.role != "ADMIN":
            return sendError("Unauthorized: Admin access required", 403)

        external_job = session.query(ExternalJob).filter(
            ExternalJob.id == job_id).first()
        if not external_job:
            return sendError("External job not found", 404)

        external_job.is_enabled = False
        session.commit()

        return sendSuccess("External job disabled successfully")

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()


class BulkJobAction(BaseModel):
    job_ids: List[int]


@router.post("/external_jobs/bulk_enable")
async def bulk_enable_external_jobs(request: Request, data: BulkJobAction):
    """Enable multiple external jobs (admin only)"""
    user_id = request.state.user["id"]

    try:
        # Check if user is admin
        user = session.query(User).filter(User.id == user_id).first()
        if not user or user.role != "ADMIN":
            return sendError("Unauthorized: Admin access required", 403)

        if not data.job_ids:
            return sendError("No job IDs provided")

        updated_count = session.query(ExternalJob).filter(
            ExternalJob.id.in_(data.job_ids)
        ).update({"is_enabled": True}, synchronize_session=False)

        session.commit()

        return sendSuccess(f"Enabled {updated_count} external jobs")

    except Exception as err:
        session.rollback()
        return sendError(str(err))
    finally:
        session.close()
