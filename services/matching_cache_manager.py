"""
Cache Manager and Optimization utilities for the job matching system
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import hashlib
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from db.connection import Base
# Base = declarative_base()


from dotenv import load_dotenv
import os

load_dotenv()

config = {
    "cache": {
        "job_match_ttl_hours": int(os.getenv("CACHE_JOB_MATCH_TTL_HOURS", 24)),
        "enable_semantic_matching": os.getenv("CACHE_ENABLE_SEMANTIC_MATCHING", "True") == "True",
        "min_cache_score": float(os.getenv("CACHE_MIN_CACHE_SCORE", 30.0))
    },
    "matching": {
        "default_min_match_score": float(os.getenv("MATCHING_DEFAULT_MIN_MATCH_SCORE", 40.0)),
        "max_recommendations": int(os.getenv("MATCHING_MAX_RECOMMENDATIONS", 50))
    }
}


@dataclass
class CacheConfig:
    """Configuration for caching behavior"""
    job_match_ttl: int = 24  # hours
    course_match_ttl: int = 72  # hours
    profile_ttl: int = 168  # 7 days
    enable_semantic_matching: bool = True
    min_cache_score: float = 30.0


class MatchCache(Base):
    """Database model for caching match results"""
    __tablename__ = "match_cache"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    # 'job_match', 'course_match', 'profile'
    cache_type = Column(String(50), nullable=False)
    data = Column(Text, nullable=False)  # JSON data
    match_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_valid = Column(Boolean, default=True)


class UserMatchingPreferences(Base):
    """Store user preferences for matching"""
    __tablename__ = "user_matching_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, unique=True, index=True)
    min_match_score = Column(Float, default=40.0)
    preferred_job_types = Column(Text)  # JSON list
    preferred_locations = Column(Text)  # JSON list
    salary_expectations = Column(Text)  # JSON object with min/max
    enable_remote_jobs = Column(Boolean, default=True)
    enable_semantic_matching = Column(Boolean, default=True)
    auto_refresh_matches = Column(Boolean, default=True)
    notification_threshold = Column(Float, default=70.0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class MatchingCacheManager:
    """Manages caching for matching operations"""

    def __init__(self, session, config: CacheConfig = None):
        self.session = session
        self.config = config or CacheConfig()

    def _generate_cache_key(self, user_id: int, item_id: int, item_type: str,
                            profile_hash: str = None) -> str:
        """Generate unique cache key for match result"""
        key_data = f"{user_id}:{item_type}:{item_id}"
        if profile_hash:
            key_data += f":{profile_hash}"

        return hashlib.sha256(key_data.encode()).hexdigest()

    def _calculate_profile_hash(self, user_profile) -> str:
        """Calculate hash of user profile for cache invalidation"""
        profile_data = {
            'skills': sorted(user_profile.skills),
            'experience_level': user_profile.experience_level,
            'resume_length': len(user_profile.resume_text) if user_profile.resume_text else 0
        }
        return hashlib.sha256(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()

    def get_cached_match(self, user_id: int, item_id: int, item_type: str,
                         profile_hash: str = None) -> Optional[Dict[str, Any]]:
        """Get cached match result if available and valid"""
        cache_key = self._generate_cache_key(
            user_id, item_id, item_type, profile_hash)

        cached = self.session.query(MatchCache).filter(
            MatchCache.cache_key == cache_key,
            MatchCache.expires_at > datetime.utcnow(),
            MatchCache.is_valid == True
        ).first()

        if cached:
            try:
                return json.loads(cached.data)
            except json.JSONDecodeError:
                # Invalid cached data, mark as invalid
                cached.is_valid = False
                self.session.commit()

        return None

    def cache_match_result(self, user_id: int, item_id: int, item_type: str,
                           match_data: Dict[str, Any], profile_hash: str = None):
        """Cache a match result"""
        cache_key = self._generate_cache_key(
            user_id, item_id, item_type, profile_hash)

        # Determine TTL based on item type
        if item_type in ['job', 'external_job']:
            ttl_hours = self.config.job_match_ttl
        elif item_type == 'course':
            ttl_hours = self.config.course_match_ttl
        else:
            ttl_hours = self.config.job_match_ttl

        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

        # Remove existing cache entry
        existing = self.session.query(MatchCache).filter(
            MatchCache.cache_key == cache_key
        ).first()

        if existing:
            existing.data = json.dumps(match_data)
            existing.match_score = match_data.get('match_score', 0.0)
            existing.expires_at = expires_at
            existing.is_valid = True
            existing.created_at = datetime.utcnow()
        else:
            cache_entry = MatchCache(
                user_id=user_id,
                cache_key=cache_key,
                cache_type=f"{item_type}_match",
                data=json.dumps(match_data),
                match_score=match_data.get('match_score', 0.0),
                expires_at=expires_at
            )
            self.session.add(cache_entry)

        self.session.commit()

    def invalidate_user_cache(self, user_id: int, cache_types: List[str] = None):
        """Invalidate all cache entries for a user"""
        query = self.session.query(MatchCache).filter(
            MatchCache.user_id == user_id)

        if cache_types:
            query = query.filter(MatchCache.cache_type.in_(cache_types))

        query.update({MatchCache.is_valid: False})
        self.session.commit()

    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        expired_count = self.session.query(MatchCache).filter(
            MatchCache.expires_at <= datetime.utcnow()
        ).delete()

        self.session.commit()
        return expired_count

    def get_cache_stats(self, user_id: int = None) -> Dict[str, Any]:
        """Get cache statistics"""
        query = self.session.query(MatchCache)

        if user_id:
            query = query.filter(MatchCache.user_id == user_id)

        total_entries = query.count()
        valid_entries = query.filter(
            MatchCache.is_valid == True,
            MatchCache.expires_at > datetime.utcnow()
        ).count()

        expired_entries = query.filter(
            MatchCache.expires_at <= datetime.utcnow()
        ).count()

        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'hit_rate': (valid_entries / max(total_entries, 1)) * 100
        }


class OptimizedMatchingService:
    """Optimized matching service with caching and batching"""

    def __init__(self, session, cache_manager: MatchingCacheManager = None,
                 llm_processor=None):
        self.session = session
        self.cache_manager = cache_manager or MatchingCacheManager(session)
        self.llm_processor = llm_processor

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user matching preferences"""
        prefs = self.session.query(UserMatchingPreferences).filter(
            UserMatchingPreferences.user_id == user_id
        ).first()

        if not prefs:
            # Create default preferences
            prefs = UserMatchingPreferences(user_id=user_id)
            self.session.add(prefs)
            self.session.commit()

        return {
            'min_match_score': prefs.min_match_score,
            'preferred_job_types': json.loads(prefs.preferred_job_types) if prefs.preferred_job_types else [],
            'preferred_locations': json.loads(prefs.preferred_locations) if prefs.preferred_locations else [],
            'salary_expectations': json.loads(prefs.salary_expectations) if prefs.salary_expectations else {},
            'enable_remote_jobs': prefs.enable_remote_jobs,
            'enable_semantic_matching': prefs.enable_semantic_matching,
            'auto_refresh_matches': prefs.auto_refresh_matches,
            'notification_threshold': prefs.notification_threshold
        }

    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]):
        """Update user matching preferences"""
        prefs = self.session.query(UserMatchingPreferences).filter(
            UserMatchingPreferences.user_id == user_id
        ).first()

        if not prefs:
            prefs = UserMatchingPreferences(user_id=user_id)
            self.session.add(prefs)

        # Update preferences
        if 'min_match_score' in preferences:
            prefs.min_match_score = preferences['min_match_score']
        if 'preferred_job_types' in preferences:
            prefs.preferred_job_types = json.dumps(
                preferences['preferred_job_types'])
        if 'preferred_locations' in preferences:
            prefs.preferred_locations = json.dumps(
                preferences['preferred_locations'])
        if 'salary_expectations' in preferences:
            prefs.salary_expectations = json.dumps(
                preferences['salary_expectations'])
        if 'enable_remote_jobs' in preferences:
            prefs.enable_remote_jobs = preferences['enable_remote_jobs']
        if 'enable_semantic_matching' in preferences:
            prefs.enable_semantic_matching = preferences['enable_semantic_matching']
        if 'auto_refresh_matches' in preferences:
            prefs.auto_refresh_matches = preferences['auto_refresh_matches']
        if 'notification_threshold' in preferences:
            prefs.notification_threshold = preferences['notification_threshold']

        prefs.updated_at = datetime.utcnow()
        self.session.commit()

        # Invalidate cache since preferences changed
        self.cache_manager.invalidate_user_cache(user_id)

    def batch_match_items(self, user_profile, items: List[Dict[str, Any]],
                          item_type: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Batch process multiple items for matching with caching"""
        results = []
        profile_hash = self.cache_manager._calculate_profile_hash(user_profile)

        # Get user preferences
        user_prefs = self.get_user_preferences(user_profile.user_id)

        for item in items:
            item_id = item.get('id')

            # Try cache first if enabled
            cached_result = None
            if use_cache:
                cached_result = self.cache_manager.get_cached_match(
                    user_profile.user_id, item_id, item_type, profile_hash
                )

            if cached_result:
                results.append(cached_result)
            else:
                # Calculate match
                try:
                    if self.llm_processor:
                        # Use semantic matching if enabled
                        use_semantic = user_prefs.get(
                            'enable_semantic_matching', True)
                        self.llm_processor.matching_engine.use_semantic_matching = use_semantic

                        match_result = self.llm_processor.matching_engine.calculate_comprehensive_match(
                            user_profile, item, item_type
                        )

                        match_data = {
                            'item_id': match_result.item_id,
                            'item_type': match_result.item_type,
                            'match_score': match_result.match_score,
                            'skill_match_count': match_result.skill_match_count,
                            'total_required_skills': match_result.total_required_skills,
                            'missing_skills': match_result.missing_skills,
                            'matched_skills': match_result.matched_skills,
                            'item_data': match_result.item_data
                        }

                        # Cache the result
                        if use_cache and match_result.match_score >= self.cache_manager.config.min_cache_score:
                            self.cache_manager.cache_match_result(
                                user_profile.user_id, item_id, item_type, match_data, profile_hash
                            )

                        results.append(match_data)

                except Exception as e:
                    print(f"Error matching item {item_id}: {e}")
                    continue

        return results

    def get_smart_recommendations(self, user_id: int, item_type: str = 'job',
                                  limit: int = 20) -> List[Dict[str, Any]]:
        """Get smart recommendations based on user preferences and behavior"""
        user_prefs = self.get_user_preferences(user_id)
        min_score = user_prefs['min_match_score']

        # Get cached valid matches first
        cached_matches = self.session.query(MatchCache).filter(
            MatchCache.user_id == user_id,
            MatchCache.cache_type == f"{item_type}_match",
            MatchCache.is_valid == True,
            MatchCache.expires_at > datetime.utcnow(),
            MatchCache.match_score >= min_score
        ).order_by(MatchCache.match_score.desc()).limit(limit * 2).all()

        recommendations = []
        for cached in cached_matches:
            try:
                match_data = json.loads(cached.data)
                recommendations.append(match_data)
            except json.JSONDecodeError:
                continue

        # Apply user preference filters
        filtered_recommendations = self._apply_preference_filters(
            recommendations, user_prefs, item_type)

        return filtered_recommendations[:limit]

    def _apply_preference_filters(self, recommendations: List[Dict[str, Any]],
                                  user_prefs: Dict[str, Any], item_type: str) -> List[Dict[str, Any]]:
        """Apply user preference filters to recommendations"""
        filtered = []

        for rec in recommendations:
            item_data = rec.get('item_data', {})

            # Job type filter
            if item_type in ['job', 'external_job'] and user_prefs.get('preferred_job_types'):
                job_type = item_data.get(
                    'job_type', item_data.get('type', '')).lower()
                if job_type not in [jt.lower() for jt in user_prefs['preferred_job_types']]:
                    continue

            # Location filter
            if user_prefs.get('preferred_locations'):
                location = item_data.get('location', '').lower()
                if not any(pref_loc.lower() in location for pref_loc in user_prefs['preferred_locations']):
                    # Check if remote work is enabled and this is a remote job
                    if not (user_prefs.get('enable_remote_jobs') and 'remote' in location):
                        continue

            # Salary filter for jobs
            if item_type in ['job', 'external_job'] and user_prefs.get('salary_expectations'):
                salary_prefs = user_prefs['salary_expectations']
                min_expected = salary_prefs.get('min', 0)
                max_expected = salary_prefs.get('max', float('inf'))

                item_salary = item_data.get(
                    'salary', item_data.get('salary_max', 0))
                if item_salary and (item_salary < min_expected or item_salary > max_expected):
                    continue

            filtered.append(rec)

        return filtered

    def schedule_auto_refresh(self, user_id: int):
        """Schedule automatic refresh based on user preferences"""
        user_prefs = self.get_user_preferences(user_id)

        if not user_prefs.get('auto_refresh_matches', True):
            return False

        # Check if we need to refresh based on cache age
        latest_cache = self.session.query(MatchCache).filter(
            MatchCache.user_id == user_id,
            MatchCache.is_valid == True
        ).order_by(MatchCache.created_at.desc()).first()

        if not latest_cache:
            return True  # No cache, need refresh

        # Refresh if cache is older than 12 hours
        cache_age = datetime.utcnow() - latest_cache.created_at
        return cache_age > timedelta(hours=12)


class PerformanceMonitor:
    """Monitor performance of matching operations"""

    def __init__(self):
        self.metrics = {
            'match_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'total_processing_time': 0,
            'average_match_time': 0
        }
        self.start_time = None

    def start_operation(self):
        """Start timing an operation"""
        self.start_time = datetime.utcnow()

    def end_operation(self, operation_type: str):
        """End timing and record metrics"""
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            self.metrics['total_processing_time'] += duration

            if operation_type == 'match_calculation':
                self.metrics['match_calculations'] += 1
                self.metrics['average_match_time'] = (
                    self.metrics['total_processing_time'] /
                    self.metrics['match_calculations']
                )
            elif operation_type == 'cache_hit':
                self.metrics['cache_hits'] += 1
            elif operation_type == 'cache_miss':
                self.metrics['cache_misses'] += 1
            elif operation_type == 'llm_call':
                self.metrics['llm_calls'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_cache_ops = self.metrics['cache_hits'] + \
            self.metrics['cache_misses']
        cache_hit_rate = (
            self.metrics['cache_hits'] / max(total_cache_ops, 1)) * 100

        return {
            **self.metrics,
            'cache_hit_rate': round(cache_hit_rate, 2),
            'total_cache_operations': total_cache_ops
        }


# Usage example and integration
class EnhancedMatchingController:
    """Main controller that orchestrates all matching operations"""

    def __init__(self, session, llm_processor=None):
        self.session = session
        self.cache_manager = MatchingCacheManager(session)
        self.optimized_service = OptimizedMatchingService(
            session, self.cache_manager, llm_processor)
        self.performance_monitor = PerformanceMonitor()
        self.llm_processor = llm_processor

    def process_user_matching_request(self, user_id: int, item_type: str = 'job',
                                      force_refresh: bool = False, limit: int = 20) -> Dict[str, Any]:
        """Process a complete user matching request with optimization"""

        self.performance_monitor.start_operation()

        try:
            # Get user profile
            from services.enhanced_matching_system import JobCourseMatchingService
            service = JobCourseMatchingService(
                self.session, self.llm_processor)
            user_profile = service.get_user_profile_from_db(user_id)

            if not user_profile:
                return {"error": "User profile not found", "recommendations": []}

            # Check if we need to refresh
            should_refresh = force_refresh or self.optimized_service.schedule_auto_refresh(
                user_id)

            all_items = []
            if should_refresh:
                # Clear cache and regenerate matches
                self.cache_manager.invalidate_user_cache(
                    user_id, [f"{item_type}_match"])

                # Get fresh data
                if item_type == 'job':
                    from models import Job, ExternalJob
                    internal_jobs = self.session.query(Job).all()
                    external_jobs = self.session.query(ExternalJob).filter(
                        ExternalJob.is_active == True, ExternalJob.is_enabled == True
                    ).all()

                    all_items = []
                    # Convert internal jobs
                    for job in internal_jobs:
                        all_items.append({
                            "id": job.id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "description": job.description,
                            "skills": job.skills or [],
                            "type": "internal_job"
                        })

                    # Convert external jobs
                    for job in external_jobs:
                        all_items.append({
                            "id": job.id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "description": job.description or "",
                            "skills": job.skills or [],
                            "type": "external_job"
                        })

                elif item_type == 'course':
                    from models import Course
                    courses = self.session.query(Course).filter(
                        Course.isActive == True).all()

                    all_items = []
                    for course in courses:
                        all_items.append({
                            "id": course.id,
                            "title": course.title,
                            "description": course.description,
                            "skills": course.skills or [],
                            "type": "course"
                        })

                # Batch process matches
                matches = self.optimized_service.batch_match_items(
                    user_profile, all_items, item_type, use_cache=True
                )

                self.performance_monitor.end_operation('match_calculation')

            # Get smart recommendations (this will use cache if available)
            recommendations = self.optimized_service.get_smart_recommendations(
                user_id, item_type, limit
            )

            return {
                "recommendations": recommendations,
                "total_count": len(recommendations),
                "cache_refreshed": should_refresh,
                "performance_stats": self.performance_monitor.get_stats()
            }

        except Exception as e:
            return {
                "error": str(e),
                "recommendations": [],
                "performance_stats": self.performance_monitor.get_stats()
            }

    def cleanup_and_optimize(self):
        """Perform cleanup and optimization tasks"""
        # Clean expired cache
        expired_count = self.cache_manager.cleanup_expired_cache()

        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_stats()

        return {
            "expired_entries_removed": expired_count,
            "cache_statistics": cache_stats,
            "performance_stats": self.performance_monitor.get_stats()
        }


# Database migration helper
def create_cache_tables(engine):
    """Create cache-related tables"""
    Base.metadata.create_all(engine)


# Configuration helper
class MatchingSystemConfig:
    """Configuration management for the matching system"""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "cache": {
                "job_match_ttl_hours": 24,
                "course_match_ttl_hours": 72,
                "profile_ttl_hours": 168,
                "min_cache_score": 30.0,
                "enable_caching": True
            },
            "matching": {
                "enable_semantic_matching": True,
                "default_min_match_score": 40.0,
                "max_recommendations": 50,
                "batch_size": 100
            },
            "performance": {
                "enable_monitoring": True,
                "log_slow_operations": True,
                "slow_operation_threshold_seconds": 5.0
            }
        }

    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables"""
        import os

        config = MatchingSystemConfig.get_default_config()

        # Override with environment variables if present
        if os.getenv("CACHE_JOB_MATCH_TTL"):
            config["cache"]["job_match_ttl_hours"] = int(
                os.getenv("CACHE_JOB_MATCH_TTL"))

        if os.getenv("ENABLE_SEMANTIC_MATCHING"):
            config["matching"]["enable_semantic_matching"] = os.getenv(
                "ENABLE_SEMANTIC_MATCHING").lower() == "true"

        if os.getenv("DEFAULT_MIN_MATCH_SCORE"):
            config["matching"]["default_min_match_score"] = float(
                os.getenv("DEFAULT_MIN_MATCH_SCORE"))

        return config
