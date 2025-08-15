"""
Simple job scraper for external job boards
Scrapes jobs from StackOverflow, We Work Remotely, and other sites
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re


class JobScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_stackoverflow_jobs(self, limit: int = 20) -> List[Dict]:
        """Scrape jobs from StackOverflow Jobs (simplified)"""
        jobs = []
        try:
            # Note: StackOverflow Jobs was discontinued, using a mock structure
            # In real implementation, you'd use their API or alternative sources
            
            # Mock data structure for demonstration
            mock_jobs = [
                {
                    'title': 'Senior Python Developer',
                    'company': 'TechCorp',
                    'location': 'Remote',
                    'description': 'Looking for experienced Python developer with Django/FastAPI experience',
                    'skills': ['Python', 'Django', 'FastAPI', 'PostgreSQL'],
                    'salary_min': 80000,
                    'salary_max': 120000,
                    'job_type': 'FULL_TIME',
                    'apply_url': 'https://stackoverflow.com/jobs/example1',
                    'source': 'StackOverflow',
                    'posted_date': datetime.now() - timedelta(days=2)
                },
                {
                    'title': 'Frontend React Developer',
                    'company': 'StartupXYZ',
                    'location': 'San Francisco, CA',
                    'description': 'React developer needed for fast-growing startup',
                    'skills': ['React', 'JavaScript', 'TypeScript', 'CSS'],
                    'salary_min': 70000,
                    'salary_max': 100000,
                    'job_type': 'FULL_TIME',
                    'apply_url': 'https://stackoverflow.com/jobs/example2',
                    'source': 'StackOverflow',
                    'posted_date': datetime.now() - timedelta(days=1)
                }
            ]
            
            return mock_jobs[:limit]
            
        except Exception as e:
            print(f"Error scraping StackOverflow: {e}")
            return []

    def scrape_weworkremotely(self, limit: int = 20) -> List[Dict]:
        """Scrape jobs from We Work Remotely"""
        jobs = []
        try:
            url = "https://weworkremotely.com/categories/remote-programming-jobs"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            job_listings = soup.find_all('li', class_='feature')[:limit]
            
            for job in job_listings:
                try:
                    title_elem = job.find('span', class_='title')
                    company_elem = job.find('span', class_='company')
                    link_elem = job.find('a')
                    
                    if title_elem and company_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        company = company_elem.get_text(strip=True)
                        job_url = f"https://weworkremotely.com{link_elem.get('href')}"
                        
                        # Extract skills from title/description (simplified)
                        skills = self._extract_skills_from_text(title)
                        
                        job_data = {
                            'title': title,
                            'company': company,
                            'location': 'Remote',
                            'description': title,  # Simplified
                            'skills': skills,
                            'salary_min': None,
                            'salary_max': None,
                            'job_type': 'FULL_TIME',
                            'apply_url': job_url,
                            'source': 'We Work Remotely',
                            'posted_date': datetime.now()
                        }
                        jobs.append(job_data)
                        
                except Exception as e:
                    print(f"Error parsing job: {e}")
                    continue
                    
            time.sleep(1)  # Be respectful to the server
            
        except Exception as e:
            print(f"Error scraping We Work Remotely: {e}")
            
        return jobs

    def scrape_remoteok(self, limit: int = 20) -> List[Dict]:
        """Scrape jobs from Remote OK"""
        jobs = []
        try:
            url = "https://remoteok.io/remote-dev-jobs"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            job_listings = soup.find_all('tr', class_='job')[:limit]
            
            for job in job_listings:
                try:
                    title_elem = job.find('h2', itemprop='title')
                    company_elem = job.find('h3', itemprop='name')
                    link_elem = job.find('a', itemprop='url')
                    
                    if title_elem and company_elem:
                        title = title_elem.get_text(strip=True)
                        company = company_elem.get_text(strip=True)
                        job_url = f"https://remoteok.io{link_elem.get('href')}" if link_elem else ""
                        
                        # Extract skills from tags
                        skill_tags = job.find_all('span', class_='tag')
                        skills = [tag.get_text(strip=True) for tag in skill_tags]
                        
                        job_data = {
                            'title': title,
                            'company': company,
                            'location': 'Remote',
                            'description': title,
                            'skills': skills,
                            'salary_min': None,
                            'salary_max': None,
                            'job_type': 'FULL_TIME',
                            'apply_url': job_url,
                            'source': 'Remote OK',
                            'posted_date': datetime.now()
                        }
                        jobs.append(job_data)
                        
                except Exception as e:
                    print(f"Error parsing Remote OK job: {e}")
                    continue
                    
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping Remote OK: {e}")
            
        return jobs

    def scrape_greenhouse_jobs(self, company_domain: str = None, limit: int = 20) -> List[Dict]:
        """Scrape jobs from Greenhouse (requires company-specific URLs)"""
        jobs = []
        try:
            # Example companies using Greenhouse
            companies = ['airbnb', 'stripe', 'coinbase'] if not company_domain else [company_domain]
            
            for company in companies:
                try:
                    url = f"https://boards.greenhouse.io/{company}"
                    response = self.session.get(url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    job_listings = soup.find_all('div', class_='opening')
                    
                    for job in job_listings[:limit//len(companies)]:
                        try:
                            title_elem = job.find('a')
                            location_elem = job.find('span', class_='location')
                            
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                location = location_elem.get_text(strip=True) if location_elem else 'Not specified'
                                job_url = f"https://boards.greenhouse.io{title_elem.get('href')}"
                                
                                skills = self._extract_skills_from_text(title)
                                
                                job_data = {
                                    'title': title,
                                    'company': company.title(),
                                    'location': location,
                                    'description': title,
                                    'skills': skills,
                                    'salary_min': None,
                                    'salary_max': None,
                                    'job_type': 'FULL_TIME',
                                    'apply_url': job_url,
                                    'source': 'Greenhouse',
                                    'posted_date': datetime.now()
                                }
                                jobs.append(job_data)
                                
                        except Exception as e:
                            print(f"Error parsing Greenhouse job: {e}")
                            continue
                            
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error scraping {company} on Greenhouse: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping Greenhouse: {e}")
            
        return jobs

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract common tech skills from job title/description"""
        common_skills = [
            'Python', 'JavaScript', 'Java', 'React', 'Node.js', 'Django', 'FastAPI',
            'PostgreSQL', 'MySQL', 'MongoDB', 'AWS', 'Docker', 'Kubernetes',
            'TypeScript', 'Vue.js', 'Angular', 'PHP', 'Ruby', 'Go', 'Rust',
            'CSS', 'HTML', 'Git', 'Linux', 'Redis', 'GraphQL', 'REST API'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in common_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
                
        return found_skills

    def scrape_all_sources(self, limit_per_source: int = 10) -> List[Dict]:
        """Scrape jobs from all sources"""
        all_jobs = []
        
        print("Scraping StackOverflow...")
        all_jobs.extend(self.scrape_stackoverflow_jobs(limit_per_source))
        
        print("Scraping We Work Remotely...")
        all_jobs.extend(self.scrape_weworkremotely(limit_per_source))
        
        print("Scraping Remote OK...")
        all_jobs.extend(self.scrape_remoteok(limit_per_source))
        
        print("Scraping Greenhouse...")
        all_jobs.extend(self.scrape_greenhouse_jobs(limit=limit_per_source))
        
        # Remove duplicates based on title and company
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = (job['title'].lower(), job['company'].lower())
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
                
        return unique_jobs


def main():
    """Test the scraper"""
    scraper = JobScraper()
    jobs = scraper.scrape_all_sources(limit_per_source=5)
    
    print(f"Scraped {len(jobs)} jobs:")
    for job in jobs:
        print(f"- {job['title']} at {job['company']} ({job['source']})")
        print(f"  Skills: {', '.join(job['skills'])}")
        print(f"  Apply: {job['apply_url']}")
        print()


if __name__ == "__main__":
    main()
