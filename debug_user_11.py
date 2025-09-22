#!/usr/bin/env python3
"""
Debug user 11 profile and matching issues
"""

from services.enhanced_matching_system import JobCourseMatchingService, GenericLLMProcessor
from models import Skill
from models.job import ExternalJob, ExternalJobMatch
from models.user import User, JobSeekerSkill, UserResume
from db.connection import session
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def debug_user_11():
    print('=== DEBUGGING USER 11 PROFILE ===')
    user_id = 11

    # Check if user exists
    user = session.query(User).filter(User.id == user_id).first()
    if user:
        print(f'User 11: Found')
        print(f'  - Email: {user.email}')
        print(f'  - Created: {user.created}')
    else:
        print('User 11: Not found')
        return

    # Check user skills
    user_skills = session.query(JobSeekerSkill).filter(
        JobSeekerSkill.user_id == user_id).all()
    print(f'\nUser 11 has {len(user_skills)} skills:')

    skill_names = []
    for us in user_skills[:15]:  # Show first 15 skills
        skill = session.query(Skill).filter(Skill.id == us.skill_id).first()
        if skill:
            skill_names.append(skill.name)
            print(f'  - {skill.name}')

    # Check user resume
    resume = session.query(UserResume).filter(
        UserResume.user_id == user_id).first()
    if resume:
        print(f'\nUser 11 resume: Found')
        print(f'  - Resume ID: {resume.id}')
        print(f'  - Filename: {resume.filename}')
        print(f'  - Created: {resume.created}')
        print(f'  - LLM Insights: {"Yes" if resume.llm_insights else "No"}')
    else:
        print('\nUser 11 resume: Not found')

    # Test user profile creation
    llm_processor = GenericLLMProcessor()
    service = JobCourseMatchingService(session, llm_processor)
    user_profile = service.get_user_profile_from_db(user_id)

    if user_profile:
        print(f'\nUser profile: Found')
        print(f'  - User ID: {user_profile.user_id}')
        print(f'  - Skills count: {len(user_profile.skills)}')
        print(f'  - First 10 skills: {user_profile.skills[:10]}')
    else:
        print('\nUser profile: Not found')

    # Check external jobs
    external_jobs = session.query(ExternalJob).filter(
        ExternalJob.is_enabled == True).all()
    jobs_with_skills = [
        job for job in external_jobs if job.skills and len(job.skills) > 0]
    print(
        f'\nExternal jobs: {len(external_jobs)} total, {len(jobs_with_skills)} with skills')

    # Check existing matches
    matches = session.query(ExternalJobMatch).filter(
        ExternalJobMatch.user_id == user_id).all()
    print(f'\nExisting matches for user 11: {len(matches)}')
    for match in matches:
        print(
            f'  - Job {match.external_job_id}: {match.match_score}% (skills: {match.skill_match_count})')

    session.close()


if __name__ == "__main__":
    debug_user_11()
