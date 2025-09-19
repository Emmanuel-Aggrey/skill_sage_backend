#!/usr/bin/env python3
"""
Script to fix missing skills in external jobs by extracting them from job descriptions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.connection import session
from models.job import ExternalJob
from services.llm import BaseLLMClient
import json


class SkillsExtractor:
    def __init__(self):
        self.llm_client = BaseLLMClient()
    
    def extract_skills_from_description(self, description: str) -> list:
        """Extract skills from job description using LLM"""
        if not description:
            return []
        
        try:
            prompt = f"""
            Extract technical skills from this job description. Return only a JSON array of skills.
            Focus on programming languages, frameworks, databases, tools, and technologies.
            
            Job Description:
            {description[:1000]}
            
            Return format: ["skill1", "skill2", "skill3"]
            """
            
            response = self.llm_client.query_llm(prompt)
            
            # Parse JSON response
            skills = json.loads(response)
            
            if isinstance(skills, list):
                return [skill.strip() for skill in skills if skill.strip()]
            
        except Exception as e:
            print(f"Error extracting skills: {e}")
        
        return []
    
    def extract_skills_from_title(self, title: str) -> list:
        """Extract skills from job title using common patterns"""
        common_skills = [
            'Python', 'JavaScript', 'Java', 'React', 'Node.js', 'Django', 'FastAPI',
            'PostgreSQL', 'MySQL', 'MongoDB', 'AWS', 'Docker', 'Kubernetes',
            'TypeScript', 'Vue.js', 'Angular', 'PHP', 'Ruby', 'Go', 'Rust',
            'CSS', 'HTML', 'Git', 'Linux', 'Redis', 'GraphQL', 'REST API',
            'Flutter', 'Dart', 'Swift', 'Kotlin', 'C++', 'C#', '.NET',
            'Spring', 'Laravel', 'Express', 'Flask', 'Celery', 'Pandas',
            'NumPy', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Elasticsearch'
        ]
        
        found_skills = []
        title_lower = title.lower()
        
        for skill in common_skills:
            if skill.lower() in title_lower:
                found_skills.append(skill)
                
        return found_skills
    
    def update_missing_skills(self):
        """Update external jobs that have missing skills"""
        print("🔄 Updating missing skills for external jobs...")
        
        # Get jobs without skills
        jobs_without_skills = session.query(ExternalJob).filter(
            (ExternalJob.skills == None) | (ExternalJob.skills == [])
        ).all()
        
        print(f"Found {len(jobs_without_skills)} jobs without skills")
        
        updated_count = 0
        for job in jobs_without_skills:
            try:
                print(f"\n📝 Processing: {job.title}")
                
                # First try to extract from title
                title_skills = self.extract_skills_from_title(job.title)
                
                # Then try to extract from description using LLM
                description_skills = []
                if job.description:
                    description_skills = self.extract_skills_from_description(job.description)
                
                # Combine and deduplicate skills
                all_skills = list(set(title_skills + description_skills))
                
                if all_skills:
                    job.skills = all_skills
                    updated_count += 1
                    print(f"✅ Updated with skills: {all_skills}")
                else:
                    print(f"❌ No skills found")
                    
            except Exception as e:
                print(f"❌ Error updating {job.title}: {e}")
                continue
        
        try:
            session.commit()
            print(f"\n✅ Successfully updated {updated_count} jobs with skills")
        except Exception as e:
            session.rollback()
            print(f"❌ Error committing changes: {e}")
        finally:
            session.close()


def main():
    extractor = SkillsExtractor()
    extractor.update_missing_skills()


if __name__ == "__main__":
    main()
