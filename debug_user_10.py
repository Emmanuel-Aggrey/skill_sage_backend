#!/usr/bin/env python3
"""
Debug script for user 10 profile and matching issues
"""

from services.enhanced_matching_system import JobCourseMatchingService, GenericLLMProcessor
from models.job import ExternalJob, ExternalJobMatch
from models.user import JobSeekerSkill, UserResume, User
from db.connection import session
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def debug_user_10():
    print('=== DEBUGGING USER 10 PROFILE ===')
    user_id = 10

    # Check if user exists
    user = session.query(User).filter(User.id == user_id).first()
    print(f'User {user_id}: {"Found" if user else "Not found"}')
    if user:
        print(f'  - Email: {user.email}')
        print(f'  - Created: {user.created}')

    # Check if user has skills
    skills = session.query(JobSeekerSkill).filter(
        JobSeekerSkill.user_id == user_id).all()
    print(f'\nUser {user_id} has {len(skills)} skills:')
    for skill in skills[:15]:  # Show first 15 skills
        print(f'  - {skill.skill.name}')

    # Check if user has resume
    resume = session.query(UserResume).filter(
        UserResume.user_id == user_id).first()
    print(f'\nUser {user_id} resume: {"Found" if resume else "Not found"}')
    if resume:
        print(f'  - Resume ID: {resume.id}')
        print(f'  - Filename: {resume.filename}')
        print(f'  - Created: {resume.created}')
        print(f'  - LLM Insights: {"Yes" if resume.llm_insights else "No"}')

    # Try to get user profile using the service
    try:
        service = JobCourseMatchingService(session, GenericLLMProcessor())
        user_profile = service.get_user_profile_from_db(user_id)
        print(f'\nUser profile: {"Found" if user_profile else "Not found"}')
        if user_profile:
            print(f'  - User ID: {user_profile.user_id}')
            print(f'  - Skills count: {len(user_profile.skills)}')
            print(f'  - First 10 skills: {user_profile.skills[:10]}')
        else:
            print('  - Profile creation failed')
    except Exception as e:
        print(f'  - Error getting profile: {e}')

    # Check external jobs with skills
    external_jobs = session.query(ExternalJob).filter(
        ExternalJob.is_enabled == True).all()
    jobs_with_skills = [
        job for job in external_jobs if job.skills and len(job.skills) > 0]
    print(
        f'\nExternal jobs: {len(external_jobs)} total, {len(jobs_with_skills)} with skills')

    # Check existing matches for user 10
    matches = session.query(ExternalJobMatch).filter(
        ExternalJobMatch.user_id == user_id).all()
    print(f'\nExisting matches for user {user_id}: {len(matches)}')
    for match in matches:
        print(
            f'  - Job {match.external_job_id}: {match.match_score}% (skills: {match.skill_match_count})')

    session.close()


if __name__ == "__main__":
    debug_user_10()
