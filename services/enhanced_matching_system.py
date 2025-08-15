import io
from PyPDF2 import PdfReader
import requests
import json
import PyPDF2
from decouple import config
from typing import Union, List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import re
import time
import random

GEMINI_API_KEY = config("GEMINI_API_KEY")
GEMINI_MIN_REQUEST_INTERVAL = config("GEMINI_MIN_REQUEST_INTERVAL", 4)


@dataclass
class MatchResult:
    """Data class for match results"""
    item_id: int
    item_type: str  # 'job', 'external_job', 'course'
    match_score: float
    skill_match_count: int
    total_required_skills: int
    missing_skills: List[str]
    matched_skills: List[str]
    item_data: Dict[str, Any]


@dataclass
class UserProfile:
    """User profile data for matching"""
    user_id: int
    skills: List[str]
    resume_text: str
    experience_level: Optional[str] = None


class TextProcessor(ABC):
    """Abstract base class for text processing utilities."""

    @abstractmethod
    def process(self, input_data: Any) -> str:
        """Process input data and return text."""
        pass


class PDFTextProcessor(TextProcessor):
    """Processor for PDF files."""

    def process(self, pdf_input: Union[str, bytes]) -> str:
        """
        Extract text from PDF.

        Args:
            pdf_input: Either file path (str) or PDF bytes (bytes)

        Returns:
            Extracted text from PDF
        """
        if isinstance(pdf_input, str):
            return self._extract_from_path(pdf_input)
        elif isinstance(pdf_input, bytes):
            return self.extract_from_bytes(pdf_input)
        else:
            raise ValueError(
                "PDF input must be either file path (str) or bytes")

    def _extract_from_path(self, pdf_path: str) -> str:
        """Extract text from PDF file path."""
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    def extract_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() or ""
        return text_content.strip()


class PlainTextProcessor(TextProcessor):
    """Processor for plain text input."""

    def process(self, text_input: str) -> str:
        """
        Process plain text input.

        Args:
            text_input: Plain text string

        Returns:
            Cleaned text
        """
        return text_input.strip()


class SmartMatchingProcessor:
    """Enhanced processor for intelligent job/course matching"""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def calculate_semantic_match(self, user_profile: UserProfile, item_description: str,
                                 item_requirements: List[str]) -> float:
        """
        Use LLM to calculate semantic match between user profile and job/course

        Args:
            user_profile: User's profile data
            item_description: Job/course description
            item_requirements: Required skills/qualifications

        Returns:
            Match score (0-100)
        """
        prompt = f"""
        Analyze the compatibility between this user profile and job/course opportunity.
        
        USER PROFILE:
        Skills: {', '.join(user_profile.skills)}
        Resume Context: {user_profile.resume_text[:1500]}...
        
        JOB/COURSE OPPORTUNITY:
        Description: {item_description}
        Requirements: {', '.join(item_requirements) if item_requirements else 'Not specified'}
        
        Please provide a compatibility score from 0-100 based on:
        1. Skill overlap and relevance
        2. Experience level match
        3. Career progression suitability
        4. Learning potential
        
        Return ONLY a number between 0-100 representing the match percentage.
        """

        try:
            response = self._query_llm(prompt)
            # Extract number from response
            score_match = re.search(r'\b(\d{1,3})\b', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0), 100)  # Clamp between 0-100
            return 0.0
        except Exception as e:
            print(f"Error calculating semantic match: {e}")
            return 0.0

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt"""

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        })

        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }

        response = requests.post(self.api_url, headers=headers, data=payload)
        response.raise_for_status()

        resp_dict = response.json()
        return resp_dict['candidates'][0]['content']['parts'][0]['text'].strip()


class AdvancedMatchingEngine:
    """Advanced matching engine with multiple algorithms"""

    def __init__(self, use_semantic_matching: bool = True):
        self.use_semantic_matching = use_semantic_matching
        self.semantic_processor = SmartMatchingProcessor() if use_semantic_matching else None

    def calculate_skill_match_score(self, user_skills: List[str],
                                    required_skills: List[str]) -> Tuple[float, int, List[str], List[str]]:
        """
        Calculate skill-based match score

        Args:
            user_skills: List of user's skills
            required_skills: List of required skills

        Returns:
            Tuple of (match_score, matched_count, missing_skills, matched_skills)
        """
        if not required_skills:
            return 100.0, 0, [], []

        # Normalize skills for comparison (case insensitive, remove extra spaces)
        normalized_user_skills = [skill.lower().strip()
                                  for skill in user_skills]
        normalized_required_skills = [
            skill.lower().strip() for skill in required_skills]

        matched_skills = []
        missing_skills = []

        for req_skill in required_skills:
            normalized_req = req_skill.lower().strip()

            # Check for exact match
            if normalized_req in normalized_user_skills:
                matched_skills.append(req_skill)
            else:
                # Check for partial matches (e.g., "React.js" vs "React")
                partial_match = any(
                    normalized_req in user_skill or user_skill in normalized_req
                    for user_skill in normalized_user_skills
                )
                if partial_match:
                    matched_skills.append(req_skill)
                else:
                    missing_skills.append(req_skill)

        match_count = len(matched_skills)
        match_score = (match_count / len(required_skills)) * 100

        return match_score, match_count, missing_skills, matched_skills

    def calculate_comprehensive_match(self, user_profile: UserProfile,
                                      item_data: Dict[str, Any],
                                      item_type: str) -> MatchResult:
        """
        Calculate comprehensive match using multiple factors

        Args:
            user_profile: User's profile
            item_data: Job/course data
            item_type: Type of item ('job', 'external_job', 'course')

        Returns:
            MatchResult object
        """
        # Extract relevant data based on item type
        if item_type == 'external_job':
            required_skills = item_data.get('skills', [])
            description = item_data.get('description', '')
            item_id = item_data.get('id')
        elif item_type == 'job':
            required_skills = item_data.get('skills', [])
            description = item_data.get('description', '')
            item_id = item_data.get('id')
        elif item_type == 'course':
            required_skills = item_data.get('skills', [])
            description = item_data.get('description', '')
            item_id = item_data.get('id')
        else:
            raise ValueError(f"Unsupported item type: {item_type}")

        # Calculate skill-based match
        skill_match_score, match_count, missing_skills, matched_skills = self.calculate_skill_match_score(
            user_profile.skills, required_skills
        )

        # Calculate semantic match if enabled
        semantic_score = 0.0
        if self.use_semantic_matching and self.semantic_processor:
            semantic_score = self.semantic_processor.calculate_semantic_match(
                user_profile, description, required_skills
            )

        # Combine scores (weighted average)
        if self.use_semantic_matching:
            final_score = (skill_match_score * 0.6) + (semantic_score * 0.4)
        else:
            final_score = skill_match_score

        return MatchResult(
            item_id=item_id,
            item_type=item_type,
            match_score=round(final_score, 2),
            skill_match_count=match_count,
            total_required_skills=len(required_skills),
            missing_skills=missing_skills,
            matched_skills=matched_skills,
            item_data=item_data
        )


class GenericLLMProcessor:
    """Enhanced Generic LLM processor with matching capabilities"""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.matching_engine = AdvancedMatchingEngine()
        self.last_request_time = None
        self.min_interval = GEMINI_MIN_REQUEST_INTERVAL

        # Register text processors
        self.processors = {
            'pdf': PDFTextProcessor(),
            'text': PlainTextProcessor()
        }

    def add_processor(self, name: str, processor: TextProcessor):
        """Add a custom text processor."""
        self.processors[name] = processor

    def process_input(self, input_data: Any, input_type: str) -> str:
        """
        Process input data based on type.

        Args:
            input_data: The input data (PDF path, bytes, or text)
            input_type: Type of input ('pdf', 'text', or custom type)

        Returns:
            Processed text
        """
        if input_type not in self.processors:
            raise ValueError(f"Unsupported input type: {input_type}")

        return self.processors[input_type].process(input_data)

    def _query_llm(self, prompt: str) -> str:
        """Enhanced _query_llm with rate limiting and retry logic"""

        # Enforce minimum delay between requests
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                print(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        # Retry logic with exponential backoff
        for attempt in range(3):  # 3 attempts max
            try:
                payload = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}]
                })

                headers = {
                    'Content-Type': 'application/json',
                    'X-goog-api-key': self.api_key
                }

                # Record request time
                self.last_request_time = time.time()

                response = requests.post(
                    self.api_url, headers=headers, data=payload, timeout=120)

                # Handle 429 specifically
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    print(
                        f"Rate limited (attempt {attempt + 1}), waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                resp_dict = response.json()
                return resp_dict['candidates'][0]['content']['parts'][0]['text'].strip()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < 2:
                    # Exponential backoff: 10s, 20s, then fail
                    wait_time = (2 ** attempt) * 10 + random.uniform(1, 5)
                    print(
                        f"Rate limit retry {attempt + 1}, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    continue
                raise e
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s delay for other errors
                    continue
                raise e

        raise Exception("Max retries exceeded for LLM request")

    # def query_llm(self, text: str, prompt_template: str = None, custom_prompt: str = None) -> str:
    #     """
    #     Send text to LLM with specified prompt.

    #     Args:
    #         text: Input text to process
    #         prompt_template: Pre-defined prompt template name
    #         custom_prompt: Custom prompt string (overrides template)

    #     Returns:
    #         Raw LLM response text
    #     """
    #     if custom_prompt:
    #         prompt_text = f"{text}\n\n{custom_prompt}"
    #     elif prompt_template:
    #         prompt_text = self._get_prompt_template(prompt_template, text)
    #     else:
    #         prompt_text = text

    #     payload = json.dumps({
    #         "contents": [
    #             {"parts": [{"text": prompt_text}]}
    #         ]
    #     })

    #     headers = {
    #         'Content-Type': 'application/json',
    #         'X-goog-api-key': self.api_key
    #     }

    #     response = requests.post(self.api_url, headers=headers, data=payload)
    #     response.raise_for_status()

    #     resp_dict = response.json()
    #     return resp_dict['candidates'][0]['content']['parts'][0]['text'].strip()

    def _get_prompt_template(self, template_name: str, text: str) -> str:
        """Get pre-defined prompt templates."""
        templates = {
            'skills_extraction': f"{text}\n\nJust list the skills of this user from this data, nothing else. Return empty list if there are no skills.",
            'summary': f"{text}\n\nProvide a concise summary of the above text.",
            'key_points': f"{text}\n\nExtract the key points from the above text as a bulleted list.",
            'sentiment': f"{text}\n\nAnalyze the sentiment of the above text (positive, negative, or neutral) and explain why.",
            'experience_level': f"{text}\n\nBased on this resume, determine the experience level: 'Entry Level', 'Mid Level', 'Senior Level', or 'Executive'. Return only the level.",
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")

        return templates[template_name]

    def parse_list_output(self, llm_output: str) -> List[str]:
        """
        Parse LLM output that's expected to be a list format.

        Args:
            llm_output: Raw LLM response

        Returns:
            List of parsed items
        """
        items = []
        for line in llm_output.splitlines():
            line = line.strip()
            # Handle various bullet formats
            if line.startswith(("*", "-", "•", "→")):
                line = line[1:].strip()
            elif line.startswith(tuple(f"{i}." for i in range(1, 100))):
                # Handle numbered lists
                line = line.split(".", 1)[1].strip() if "." in line else line

            if line:  # Skip empty lines
                items.append(line)
        return items

    def parse_json_output(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse LLM output that's expected to be JSON format.

        Args:
            llm_output: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to find JSON in the response
            start_idx = llm_output.find('{')
            end_idx = llm_output.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = llm_output[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")

    def extract_text_only(self, input_data: Any, input_type: str) -> str:
        """
        Extract text from input without LLM processing.

        Args:
            input_data: Input data (PDF path, bytes, or text)
            input_type: Type of input ('pdf', 'text')

        Returns:
            Extracted raw text
        """
        return self.process_input(input_data, input_type)

    def create_user_profile(self, user_id: int, cv_input: Any, input_type: str = 'pdf') -> UserProfile:
        """
        Create comprehensive user profile from CV

        Args:
            user_id: User ID
            cv_input: CV data (PDF bytes, file path, or text)
            input_type: Type of input ('pdf', 'text')

        Returns:
            UserProfile object
        """
        # Extract text and skills
        resume_text = self.extract_text_only(cv_input, input_type)
        skills = self.process_and_extract(
            cv_input, input_type, 'list', 'skills_extraction'
        )

        # Get experience level
        experience_level = self.process_and_extract(
            cv_input, input_type, 'raw', 'experience_level'
        )

        return UserProfile(
            user_id=user_id,
            skills=skills if isinstance(skills, list) else [],
            resume_text=resume_text,
            experience_level=experience_level.strip() if experience_level else None
        )

    def find_matching_items(self, user_profile: UserProfile,
                            items: List[Dict[str, Any]],
                            item_type: str,
                            min_match_score: float = 40.0) -> List[MatchResult]:
        """
        Find matching jobs/courses for user

        Args:
            user_profile: User's profile
            items: List of jobs/courses to match against
            item_type: Type of items ('job', 'external_job', 'course')
            min_match_score: Minimum match score threshold

        Returns:
            List of MatchResult objects sorted by score (descending)
        """
        matches = []

        for item in items:
            try:
                match_result = self.matching_engine.calculate_comprehensive_match(
                    user_profile, item, item_type
                )

                if match_result.match_score >= min_match_score:
                    matches.append(match_result)

            except Exception as e:
                print(f"Error matching item {item.get('id', 'unknown')}: {e}")
                continue

        # Sort by match score (descending)
        matches.sort(key=lambda x: x.match_score, reverse=True)
        return matches

    def process_and_extract(self, input_data: Any, input_type: str,
                            output_format: str = 'raw',
                            prompt_template: str = None,
                            custom_prompt: str = None) -> Union[str, List[str], Dict[str, Any]]:
        """
        Complete pipeline: process input, query LLM, and parse output.

        Args:
            input_data: Input data (PDF path, bytes, or text)
            input_type: Type of input ('pdf', 'text')
            output_format: Expected output format ('raw', 'list', 'json')
            prompt_template: Pre-defined prompt template name (optional)
            custom_prompt: Custom prompt string (optional, overrides template)

        Returns:
            Processed result in specified format
        """
        # Step 1: Process input to text
        text = self.process_input(input_data, input_type)

        # Step 2: Query LLM
        llm_response = self.query_llm(text, prompt_template, custom_prompt)

        # Step 3: Parse output based on format
        if output_format == 'raw':
            return llm_response
        elif output_format == 'list':
            return self.parse_list_output(llm_response)
        elif output_format == 'json':
            return self.parse_json_output(llm_response)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Convenience class that maintains backward compatibility
class PDFSkillExtractor(GenericLLMProcessor):
    """Backward compatible PDF skill extractor."""

    def extract_skills_from_pdf(self, pdf_input: Union[str, bytes]) -> List[str]:
        """
        Extract skills from PDF (maintains original interface).

        Args:
            pdf_input: PDF file path or bytes

        Returns:
            List of extracted skills
        """
        return self.process_and_extract(
            input_data=pdf_input,
            input_type='pdf',
            output_format='list',
            prompt_template='skills_extraction'
        )


class JobCourseMatchingService:
    """Service class for job/course matching operations"""

    def __init__(self, session, llm_processor: GenericLLMProcessor = None):
        self.session = session
        self.llm = llm_processor or GenericLLMProcessor()

    def get_user_profile_from_db(self, user_id: int) -> Optional[UserProfile]:
        """
        Get user profile from database

        Args:
            user_id: User ID

        Returns:
            UserProfile object or None if not found
        """
        from models import UserResume, JobSeekerSkill, Skill  # Adjust import as needed

        # Get resume text
        user_resume = self.session.query(UserResume).filter(
            UserResume.user_id == user_id
        ).first()

        if not user_resume:
            return None

        # Get user skills
        user_skills_query = self.session.query(Skill.name).join(
            JobSeekerSkill, Skill.id == JobSeekerSkill.skill_id
        ).filter(JobSeekerSkill.user_id == user_id)

        skills = [skill.name for skill in user_skills_query.all()]

        return UserProfile(
            user_id=user_id,
            skills=skills,
            resume_text=user_resume.resume_text
        )

    def save_job_matches(self, user_id: int, matches: List[MatchResult]):
        """Save job matches to database"""
        from models.job import JobMatch, ExternalJobMatch  # Adjust import as needed

        for match in matches:
            try:
                if match.item_type == 'job':
                    # Check if match already exists
                    existing = self.session.query(JobMatch).filter(
                        JobMatch.user_id == user_id,
                        JobMatch.job_id == match.item_id
                    ).first()

                    if existing:
                        # Update existing match
                        existing.match_score = match.match_score
                        existing.skill_match_count = match.skill_match_count
                        existing.missing_skills = match.missing_skills
                    else:
                        # Create new match
                        job_match = JobMatch(
                            user_id=user_id,
                            job_id=match.item_id,
                            match_score=match.match_score,
                            skill_match_count=match.skill_match_count,
                            missing_skills=match.missing_skills
                        )
                        self.session.add(job_match)

                elif match.item_type == 'external_job':
                    # Check if match already exists
                    existing = self.session.query(ExternalJobMatch).filter(
                        ExternalJobMatch.user_id == user_id,
                        ExternalJobMatch.external_job_id == match.item_id
                    ).first()

                    if existing:
                        # Update existing match
                        existing.match_score = match.match_score
                        existing.skill_match_count = match.skill_match_count
                        existing.missing_skills = match.missing_skills
                    else:
                        # Create new match
                        external_job_match = ExternalJobMatch(
                            user_id=user_id,
                            external_job_id=match.item_id,
                            match_score=match.match_score,
                            skill_match_count=match.skill_match_count,
                            missing_skills=match.missing_skills
                        )
                        self.session.add(external_job_match)

            except Exception as e:
                print(f"Error saving match for item {match.item_id}: {e}")
                continue

        self.session.commit()

    def get_recommended_jobs(self, user_id: int, min_match_score: float = 40.0,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recommended jobs for user

        Args:
            user_id: User ID
            min_match_score: Minimum match score
            limit: Maximum number of results

        Returns:
            List of recommended jobs with match info
        """
        from models import JobMatch, Job, ExternalJobMatch, ExternalJob  # Adjust import as needed

        # Get job matches
        job_matches = self.session.query(JobMatch).join(
            Job, JobMatch.job_id == Job.id
        ).filter(
            JobMatch.user_id == user_id,
            JobMatch.match_score >= min_match_score
        ).order_by(JobMatch.match_score.desc()).limit(limit).all()

        # Get external job matches
        external_matches = self.session.query(ExternalJobMatch).join(
            ExternalJob, ExternalJobMatch.external_job_id == ExternalJob.id
        ).filter(
            ExternalJobMatch.user_id == user_id,
            ExternalJobMatch.match_score >= min_match_score,
            ExternalJob.is_active == True
        ).order_by(ExternalJobMatch.match_score.desc()).limit(limit).all()

        recommendations = []

        # Process job matches
        for match in job_matches:
            job_data = {
                "id": match.job.id,
                "title": match.job.title,
                "company": match.job.company,
                "location": match.job.location,
                "description": match.job.description,
                "salary": match.job.salary,
                "type": match.job.type,
                "skills": match.job.skills,
                "match_score": match.match_score,
                "skill_match_count": match.skill_match_count,
                "missing_skills": match.missing_skills,
                "job_type": "internal",
                "apply_url": None
            }
            recommendations.append(job_data)

        # Process external job matches
        for match in external_matches:
            job_data = {
                "id": match.external_job.id,
                "title": match.external_job.title,
                "company": match.external_job.company,
                "location": match.external_job.location,
                "description": match.external_job.description,
                "salary_min": match.external_job.salary_min,
                "salary_max": match.external_job.salary_max,
                "job_type": match.external_job.job_type,
                "skills": match.external_job.skills,
                "match_score": match.match_score,
                "skill_match_count": match.skill_match_count,
                "missing_skills": match.missing_skills,
                "source": match.external_job.source,
                "apply_url": match.external_job.apply_url
            }
            recommendations.append(job_data)

        # Sort all recommendations by match score
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)

        return recommendations[:limit]
