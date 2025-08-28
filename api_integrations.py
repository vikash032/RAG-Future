# api_integrations.py
import requests
import arxiv
from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime, timedelta
from .config import settings

class APIIntegrations:
    @staticmethod
    def fetch_arxiv_papers(query: str, max_results: int = None) -> List[Dict]:
        """Fetch real papers from Arxiv API"""
        if max_results is None:
            max_results = settings.max_results
            
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "doi": result.doi if hasattr(result, 'doi') else None,
                    "journal_ref": result.journal_ref if hasattr(result, 'journal_ref') else None,
                    "comment": result.comment if hasattr(result, 'comment') else None,
                    "primary_category": result.primary_category if hasattr(result, 'primary_category') else None,
                    "categories": result.categories if hasattr(result, 'categories') else []
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            st.error(f"❌ Arxiv API error: {e}")
            # Fallback to mock data
            return APIIntegrations._get_mock_arxiv_papers(query, max_results)
    
    @staticmethod
    def _get_mock_arxiv_papers(query: str, max_results: int) -> List[Dict]:
        """Fallback mock data for Arxiv"""
        papers = []
        for i in range(min(max_results, 3)):
            papers.append({
                "title": f"Research Paper {i+1}: {query.title()} in AI Systems",
                "authors": ["Dr. AI Researcher", "Prof. ML Expert"],
                "published": (datetime.now() - timedelta(days=30*i)).strftime("%Y-%m-%d"),
                "summary": f"This paper explores {query} in the context of artificial intelligence and machine learning. The research presents novel approaches and methodologies...",
                "pdf_url": f"https://arxiv.org/pdf/{1234+i}.{56789+i}",
                "entry_id": f"{1234+i}.{56789+i}",
                "categories": ["cs.AI", "cs.LG"],
                "doi": f"10.48550/arXiv.{1234+i}.{56789+i}"
            })
        return papers
    
    @staticmethod
    def fetch_clinical_trials(condition: str = "cancer", max_results: int = None) -> List[Dict]:
        """Fetch real clinical trials from ClinicalTrials.gov API"""
        if max_results is None:
            max_results = settings.max_results
            
        try:
            # ClinicalTrials.gov API endpoint
            base_url = "https://clinicaltrials.gov/api/query/full_studies"
            params = {
                "expr": condition,
                "min_rnk": 1,
                "max_rnk": max_results,
                "fmt": "json"
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                trials = []
                
                for study in data.get('FullStudiesResponse', {}).get('FullStudies', []):
                    study_data = study.get('Study', {})
                    protocol_data = study_data.get('ProtocolSection', {})
                    identification = protocol_data.get('IdentificationModule', {})
                    status = protocol_data.get('StatusModule', {})
                    design = protocol_data.get('DesignModule', {})
                    contacts = protocol_data.get('ContactsLocationsModule', {})
                    
                    trial = {
                        "id": identification.get('NCTId', 'N/A'),
                        "title": identification.get('BriefTitle', 'N/A'),
                        "status": status.get('OverallStatus', 'N/A'),
                        "sponsor": identification.get('OrgStudyIdInfo', {}).get('OrgStudyId', 'N/A'),
                        "start_date": status.get('StartDateStruct', {}).get('StartDate', 'N/A'),
                        "completion_date": status.get('CompletionDateStruct', {}).get('CompletionDate', 'N/A'),
                        "study_type": design.get('StudyType', 'N/A'),
                        "phase": design.get('PhaseList', {}).get('Phase', ['N/A'])[0] if design.get('PhaseList', {}).get('Phase') else 'N/A',
                        "enrollment": design.get('EnrollmentInfo', {}).get('EnrollmentCount', 'N/A'),
                        "intervention": ', '.join([i.get('InterventionName', '') for i in design.get('InterventionList', {}).get('Intervention', [])]),
                        "locations": [loc.get('LocationFacility', '') for loc in contacts.get('LocationList', {}).get('Location', [])],
                        "text": f"Clinical Trial: {identification.get('BriefTitle', 'N/A')} - {identification.get('BriefSummary', 'N/A')}"
                    }
                    trials.append(trial)
                
                return trials
            else:
                st.warning(f"⚠️ Clinical Trials API returned status {response.status_code}")
                return APIIntegrations._get_mock_clinical_trials(condition, max_results)
                
        except Exception as e:
            st.error(f"❌ Clinical Trials API error: {e}")
            return APIIntegrations._get_mock_clinical_trials(condition, max_results)
    
    @staticmethod
    def _get_mock_clinical_trials(condition: str, max_results: int) -> List[Dict]:
        """Fallback mock data for clinical trials"""
        trials = []
        statuses = ["Recruiting", "Active", "Completed", "Enrolling"]
        
        for i in range(min(max_results, 4)):
            trials.append({
                "id": f"NCT{123456 + i}",
                "title": f"Study of {condition.title()} Treatment - Phase {(i % 3) + 1}",
                "status": statuses[i % len(statuses)],
                "sponsor": f"Medical Center {i+1}",
                "start_date": (datetime.now() - timedelta(days=365*i)).strftime("%Y-%m-%d"),
                "completion_date": (datetime.now() + timedelta(days=365*(2-i))).strftime("%Y-%m-%d"),
                "enrollment": (100 + i*50),
                "locations": [f"City {i+1}", f"City {i+2}"],
                "intervention": f"Novel {condition} therapy approach",
                "text": f"Clinical Trial: {condition.title()} Treatment Study - This randomized controlled trial investigates novel therapeutic approaches for {condition} treatment."
            })
        return trials

# Global instance
api_client = APIIntegrations()
