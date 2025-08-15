
from db.connection import Base

from sqlalchemy.orm import mapped_column, relationship, Mapped
from sqlalchemy import (
    String, ARRAY, Boolean, JSON, Enum, ForeignKey,
    Integer, Date, Float, DateTime
)
from datetime import datetime


class JobType(Enum):
    PART_TIME = "PART_TIME"
    FULL_TIME = "FULL_TIME"


class Job(Base):
    __tablename__ = "jobs"
    id = mapped_column(Integer(), primary_key=True, nullable=False)
    title = mapped_column(String(), nullable=False)
    location = mapped_column(String(), nullable=False)
    expiry = mapped_column(Date(), nullable=False)
    salary = mapped_column(Float(), nullable=False)
    company = mapped_column(String(), nullable=False)
    description = mapped_column(String(), nullable=False)
    requirements = mapped_column(ARRAY(String()))
    image = mapped_column(String())
    type = mapped_column(String(), nullable=False)
    position = mapped_column(String(), nullable=False)
    skills = mapped_column(ARRAY(String()))
    user_id = mapped_column(ForeignKey("users.id"))


class Bookmark(Base):
    __tablename__ = "bookmarks"
    id = mapped_column(Integer(), primary_key=True, nullable=False)
    user_id = mapped_column(Integer(), ForeignKey("users.id"), nullable=False)
    job_id = mapped_column(Integer(), ForeignKey("jobs.id"), nullable=False)


class JobApplication(Base):
    __tablename__ = "job_applications"
    id = mapped_column(Integer(), primary_key=True, index=True)
    user_id = mapped_column(Integer(), ForeignKey("users.id"), nullable=False)
    job_id = mapped_column(Integer(), ForeignKey(
        "jobs.id"), nullable=False, unique=True)
    # e.g., "pending," "accepted," "rejected"
    status = mapped_column(String(), nullable=False)
    # Add more fields as needed


class Course(Base):
    __tablename__ = "courses"
    user_id = mapped_column(ForeignKey("users.id"))
    title = mapped_column(String(), nullable=False)
    sub_title = mapped_column(String(), nullable=False)
    description = mapped_column(String(), nullable=False)
    language = mapped_column(String())
    requirements = mapped_column(ARRAY(String()), default=[])
    lessons = mapped_column(ARRAY(String()))
    skills = mapped_column(ARRAY(String()))
    image = mapped_column(String())
    isActive = mapped_column(Boolean(), default=False)
    items = relationship("CourseItem", back_populates="course")


class CourseItem(Base):
    __tablename__ = "course_items"
    course_id = mapped_column(ForeignKey("courses.id"))
    name = mapped_column(String(), nullable=False)
    course = relationship("Course", back_populates="items")
    sessions = relationship("CourseSession", back_populates="item")


class CourseSession(Base):
    __tablename__ = "course_sessions"
    item_id = mapped_column(ForeignKey("course_items.id"))
    name = mapped_column(String(), nullable=False)
    video = mapped_column(String(), nullable=False)
    time = mapped_column(String())
    item = relationship("CourseItem", back_populates="sessions")


# New models for job recommendations
class JobMatch(Base):
    __tablename__ = "job_matches"
    id = mapped_column(Integer(), primary_key=True, nullable=False)
    user_id = mapped_column(Integer(), ForeignKey("users.id"), nullable=False)
    job_id = mapped_column(Integer(), ForeignKey("jobs.id"), nullable=False)
    match_score = mapped_column(Float(), nullable=False)
    skill_match_count = mapped_column(Integer(), nullable=False)
    missing_skills = mapped_column(ARRAY(String()))
    created = mapped_column(DateTime(), nullable=False,
                            default=datetime.utcnow)


class UserJobPreferences(Base):
    __tablename__ = "user_job_preferences"
    id = mapped_column(Integer(), primary_key=True, nullable=False)
    user_id = mapped_column(Integer(), ForeignKey("users.id"), nullable=False)
    preferred_locations = mapped_column(ARRAY(String()))
    preferred_job_types = mapped_column(ARRAY(String()))
    min_salary = mapped_column(Float())
    max_salary = mapped_column(Float())
    remote_ok = mapped_column(Boolean(), default=False)
    created = mapped_column(DateTime(), nullable=False,
                            default=datetime.utcnow)
    updated = mapped_column(DateTime(), onupdate=datetime.utcnow)


# External jobs from scraped sources
class ExternalJob(Base):
    __tablename__ = "external_jobs"
    id = mapped_column(Integer(), primary_key=True, nullable=False)
    title = mapped_column(String(), nullable=False)
    company = mapped_column(String(), nullable=False)
    location = mapped_column(String(), nullable=False)
    description = mapped_column(String())
    skills = mapped_column(ARRAY(String()))
    salary_min = mapped_column(Float())
    salary_max = mapped_column(Float())
    job_type = mapped_column(String(), nullable=False)
    apply_url = mapped_column(String(), nullable=False)
    # 'StackOverflow', 'We Work Remotely', etc.
    source = mapped_column(String(), nullable=False)
    external_id = mapped_column(String())  # Original job ID from source
    posted_date = mapped_column(DateTime())
    scraped_date = mapped_column(
        DateTime(), nullable=False, default=datetime.utcnow)
    is_active = mapped_column(Boolean(), default=True)
    # Admin must enable jobs
    is_enabled = mapped_column(Boolean(), default=False)


# External job matches for recommendations
class ExternalJobMatch(Base):
    __tablename__ = "external_job_matches"
    id = mapped_column(Integer(), primary_key=True, nullable=False)
    user_id = mapped_column(Integer(), ForeignKey("users.id"), nullable=False)
    external_job_id = mapped_column(
        Integer(), ForeignKey("external_jobs.id"), nullable=False)
    match_score = mapped_column(Float(), nullable=False)
    skill_match_count = mapped_column(Integer(), nullable=False)
    missing_skills = mapped_column(ARRAY(String()))
    created = mapped_column(DateTime(), nullable=False,
                            default=datetime.utcnow)
