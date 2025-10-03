#!/usr/bin/env python3
"""
AI Phone Review Engine - Cloud Deployment Version
Optimized for Streamlit Cloud deployment
"""

# CRITICAL: Page config must be FIRST
import streamlit as st
st.set_page_config(
    page_title="AI Phone Review Engine",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import requests
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import re
import os
from collections import defaultdict, Counter

# Graceful imports with fallbacks
PLOTLY_AVAILABLE = False
WORDCLOUD_AVAILABLE = False
NLTK_AVAILABLE = False
TEXTBLOB_AVAILABLE = False
BS4_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.sidebar.warning("⚠️ Plotly not available")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    pass

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
    try:
        # Check if resources are available (may fail in some cloud envs)
        nltk.data.find('vader_lexicon')
    except LookupError:
        # Download resources if not found
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    pass

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    pass

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .review-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .db-badge {
        background: #4CAF50;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .web-badge {
        background: #2196F3;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

@dataclass
class ReviewData:
    """Review data structure"""
    content: str
    rating: Optional[float] = None
    date: Optional[str] = None
    source: str = ""
    sentiment: float = 0.0
    aspects: Dict[str, float] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    credibility: float = 0.5
    is_spam: bool = False
    from_database: bool = False

@dataclass
class PhoneAnalysis:
    """Complete phone analysis results"""
    phone_name: str
    total_reviews: int
    avg_rating: Optional[float]
    sentiment_distribution: Dict[str, float]
    aspect_sentiments: Dict[str, Dict[str, int]]
    top_pros: List[str]
    top_cons: List[str]
    credibility_score: float
    spam_percentage: float
    emotional_profile: Dict[str, float]
    user_quotes: List[str]
    recommendation_score: float
    sources: List[str]
    db_reviews_count: int = 0
    web_reviews_count: int = 0

@st.cache_resource
def load_database_df(db_path: str = "final_dataset_streamlined_clean.csv") -> Tuple[pd.DataFrame, str]:
    """Load database from CSV and cache the DataFrame."""
    try:
        if os.path.exists(db_path):
            df = pd.read_csv(db_path)
            
            # Data Cleaning (as in original load_database)
            if 'rating' in df.columns:
                def parse_rating(val):
                    if pd.isna(val) or str(val).strip() == '########':
                        return None
                    try:
                        rating = float(val)
                        return rating if rating <= 5 else rating / 2
                    except (ValueError, TypeError):
                        return None
                
                df['rating_numeric'] = df['rating'].apply(parse_rating)
            
            # Ensure product column exists
            if 'product' not in df.columns:
                for alt_col in ['phone_name', 'model', 'device_name', 'phone']:
                    if alt_col in df.columns:
                        df['product'] = df[alt_col]
                        break
                else:
                    return pd.DataFrame(), "Missing product column"
            
            status = f"Loaded {len(df)} reviews"
            return df, status
        
        else:
            status = f"File not found"
            return pd.DataFrame(), status
            
    except Exception as e:
        status = f"Error: {str(e)}"
        return pd.DataFrame(), status


class DatabaseManager:
    """Manages local database operations"""
    
    def __init__(self, db_path: str = "final_dataset_streamlined_clean.csv"):
        self.db_path = db_path
        # Load data using the cached function
        self.df, self.load_status = load_database_df(db_path)
        
        if self.df.empty and 'File not found' in self.load_status:
             st.sidebar.info("ℹ️ No database file - using web search only")
        elif self.df.empty:
             st.sidebar.error(f"Database error: {self.load_status}")
        else:
             st.sidebar.success(f"✅ Database: {len(self.df)} reviews")
    
    def search_phone_reviews(self, phone_name: str) -> pd.DataFrame:
        """Search for phone reviews in database"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        phone_lower = phone_name.lower()
        # Use str.contains with regex=False for simple substring matching
        matches = self.df[
            self.df['product'].str.lower().str.contains(phone_lower, na=False, regex=False)
        ]
        
        # Original fallback logic
        if matches.empty and 'brand' in self.df.columns:
            brands = ['apple', 'samsung', 'google', 'oneplus', 'xiaomi', 'oppo', 'vivo']
            for brand in brands:
                if brand in phone_lower:
                    keywords = [w for w in phone_lower.split() if len(w) > 2 and w != brand]
                    if keywords:
                        brand_matches = self.df[
                            self.df['brand'].str.lower().str.contains(brand, na=False, regex=False)
                        ]
                        for keyword in keywords:
                            temp = brand_matches[
                                brand_matches['product'].str.lower().str.contains(keyword, na=False, regex=False)
                            ]
                            if not temp.empty:
                                matches = temp
                                break
                        break
        
        return matches

class HybridReviewEngine:
    """Main engine - checks database first, then web scrapes if needed"""
    
    def __init__(self):
        # NOTE: requests.Session is NOT CACHED, so it must be initialized here
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.db_manager = DatabaseManager()
        
        # Get API keys from Streamlit secrets or environment
        try:
            # Use st.secrets.get safely
            self.serpapi_key = st.secrets.get("SERPAPI_KEY", os.getenv('SERPAPI_KEY'))
            self.bing_key = st.secrets.get("BING_SEARCH_API_KEY", os.getenv('BING_SEARCH_API_KEY'))
        except:
            self.serpapi_key = os.getenv('SERPAPI_KEY')
            self.bing_key = os.getenv('BING_SEARCH_API_KEY')
        
        self.aspect_keywords = {
            'camera': ['camera', 'photo', 'picture', 'video', 'lens', 'zoom'],
            'battery': ['battery', 'charge', 'charging', 'power', 'life'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'processor'],
            'display': ['display', 'screen', 'brightness', 'color', 'oled'],
            'build': ['build', 'design', 'quality', 'premium', 'material'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth'],
            'software': ['software', 'android', 'ios', 'update', 'ui'],
            'sound': ['sound', 'audio', 'speaker', 'headphone', 'music']
        }
        
        self.emotion_keywords = {
            'love': ['love', 'amazing', 'excellent', 'fantastic', 'awesome'],
            'satisfaction': ['satisfied', 'happy', 'pleased', 'content', 'good'],
            'disappointment': ['disappointed', 'bad', 'terrible', 'awful'],
            'frustration': ['frustrated', 'annoying', 'irritating', 'problems'],
            'excitement': ['excited', 'thrilled', 'impressed', 'wow', 'stunning'],
            'regret': ['regret', 'mistake', 'waste', 'wrong choice'],
            'trust': ['trust', 'reliable', 'dependable', 'consistent']
        }
        
        self.min_reviews_threshold = 50

    def analyze_phone(self, phone_name: str, force_web_search: bool = False) -> PhoneAnalysis:
        """Main analysis function"""
        st.info(f"🔍 Searching for: **{phone_name}**")
        
        db_reviews = []
        web_reviews = []
        
        # Check database first
        if not force_web_search:
            with st.spinner("Checking database..."):
                db_data = self.db_manager.search_phone_reviews(phone_name)
                
                if not db_data.empty:
                    st.success(f"Found {len(db_data)} reviews in database")
                    db_reviews = self._convert_db_to_reviews(db_data)
                else:
                    st.warning("No reviews in database")
        
        # Web scrape if needed
        total_existing = len(db_reviews)
        if total_existing < self.min_reviews_threshold:
            needed = self.min_reviews_threshold - total_existing
            st.warning(f"Need {needed} more reviews - searching web...")
            
            with st.spinner(f"Scraping web for reviews..."):
                try:
                    web_reviews = self._web_scrape_reviews(phone_name, min_count=needed)
                    if web_reviews:
                        st.success(f"Scraped {len(web_reviews)} reviews")
                except Exception as e:
                    st.error(f"Web scraping error: {e}")
                    web_reviews = []
        
        # Combine reviews
        all_reviews = db_reviews + web_reviews
        
        if not all_reviews:
            return self._empty_analysis(phone_name)
        
        st.info(f"📊 Total: **{len(all_reviews)}** (DB: {len(db_reviews)}, Web: {len(web_reviews)})")
        
        # Perform analysis
        with st.spinner("Analyzing reviews..."):
            analysis = self._perform_analysis(phone_name, all_reviews)
            analysis.db_reviews_count = len(db_reviews)
            analysis.web_reviews_count = len(web_reviews)
        
        return analysis

    def _convert_db_to_reviews(self, df: pd.DataFrame) -> List[ReviewData]:
        """Convert database DataFrame to ReviewData objects"""
        reviews = []
        
        for _, row in df.iterrows():
            text = None
            for col_name in ['review_text', 'review', 'text', 'content']:
                if col_name in row and pd.notna(row[col_name]):
                    text = str(row[col_name])
                    break
            
            if not text or len(text) < 10:
                continue
            
            rating = None
            if 'rating_numeric' in row and pd.notna(row['rating_numeric']):
                rating = row['rating_numeric']
            
            review = ReviewData(
                content=text,
                rating=rating,
                date=str(row.get('date', datetime.now().strftime('%Y-%m-%d'))),
                source=str(row.get('source', 'database')),
                from_database=True
            )
            
            self._analyze_review(review)
            reviews.append(review)
        
        return reviews

    def _web_scrape_reviews(self, phone_name: str, min_count: int = 50) -> List[ReviewData]:
        """Web scrape reviews"""
        all_reviews = []
        attempts = 0
        max_attempts = 5
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        search_queries = [
            f"{phone_name} review",
            f"{phone_name} user review",
            f"{phone_name} customer feedback",
            f"{phone_name} pros cons"
        ]
        
        while len(all_reviews) < min_count and attempts < max_attempts:
            attempts += 1
            query = search_queries[(attempts - 1) % len(search_queries)]
            
            status_text.text(f"Attempt {attempts}: '{query}'...")
            
            try:
                search_results = self._search_reviews(query, max_results=20)
                new_reviews = self._analyze_search_results(search_results, phone_name)
                
                if new_reviews:
                    all_reviews.extend(new_reviews)
                    all_reviews = self._deduplicate_reviews(all_reviews)
                    
                    progress = min((len(all_reviews) / min_count) * 100, 100)
                    progress_bar.progress(int(progress) / 100)
                    status_text.text(f"Found {len(all_reviews)}/{min_count}...")
                
                if len(all_reviews) >= min_count:
                    break
                
                time.sleep(1)
                
            except Exception as e:
                st.warning(f"Attempt {attempts} failed: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return all_reviews[:min_count]

    def _deduplicate_reviews(self, reviews: List[ReviewData]) -> List[ReviewData]:
        """Remove duplicates"""
        unique_reviews = []
        seen_contents = set()
        
        for review in reviews:
            fingerprint = review.content.lower()[:150].strip()
            fingerprint = re.sub(r'\s+', ' ', fingerprint)
            
            if fingerprint not in seen_contents and len(fingerprint) > 20:
                seen_contents.add(fingerprint)
                unique_reviews.append(review)
        
        return unique_reviews

    def _search_reviews(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search for reviews"""
        if self.serpapi_key:
            return self._search_serpapi(query, max_results)
        else:
            return self._search_fallback(query, max_results)

    def _search_serpapi(self, query: str, max_results: int) -> List[Dict]:
        """Search using SerpAPI"""
        try:
            url = "https://serpapi.com/search.json"
            params = {
                'api_key': self.serpapi_key,
                'q': query,
                'num': min(max_results, 100),
                'engine': 'google'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get('organic_results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'source': urlparse(result.get('link', '')).netloc
                })
            
            return results
        except Exception as e:
            return []

    def _search_fallback(self, query: str, max_results: int) -> List[Dict]:
        """Fallback search (simplified)"""
        st.info("Using fallback search - results may be limited")
        return []

    def _analyze_search_results(self, search_results: List[Dict], phone_name: str) -> List[ReviewData]:
        """Extract reviews from search results"""
        reviews = []
        
        for result in search_results:
            content = f"{result.get('title', '')}. {result.get('snippet', '')}".strip()
            
            if (len(content) < 30 or phone_name.lower() not in content.lower()):
                continue
            
            review = ReviewData(
                content=content,
                source=result.get('source', ''),
                date=datetime.now().strftime('%Y-%m-%d'),
                from_database=False
            )
            
            self._analyze_search_snippet(review, content)
            reviews.append(review)
        
        return reviews

    def _analyze_search_snippet(self, review: ReviewData, content: str):
        """Analyze search snippet"""
        text = content.lower()
        
        # Basic sentiment
        review.sentiment = self._basic_sentiment(text)
        
        # Extract rating
        rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|/)?\s*(?:5|10)', text)
        if rating_match:
            try:
                rating = float(rating_match.group(1))
                review.rating = min(rating if rating <= 5 else rating / 2, 5.0)
            except:
                pass
        
        # Aspect analysis
        for aspect, keywords in self.aspect_keywords.items():
            score = 0
            count = 0
            for keyword in keywords:
                if keyword in text:
                    count += 1
                    score += review.sentiment
            if count > 0:
                review.aspects[aspect] = score / count
        
        review.credibility = 0.6
        review.is_spam = any(x in text for x in ['buy now', 'click here', 'deal'])

    def _basic_sentiment(self, text: str) -> float:
        """Basic sentiment analysis"""
        pos = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect']
        neg = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor']
        
        pos_count = sum(1 for w in pos if w in text)
        neg_count = sum(1 for w in neg if w in text)
        total = pos_count + neg_count
        
        return (pos_count - neg_count) / total if total > 0 else 0.0

    def _analyze_review(self, review: ReviewData):
        """Comprehensive review analysis"""
        text = review.content.lower()
        
        # Sentiment
        if TEXTBLOB_AVAILABLE:
            try:
                review.sentiment = TextBlob(review.content).sentiment.polarity
            except:
                review.sentiment = self._basic_sentiment(text)
        else:
            review.sentiment = self._basic_sentiment(text)
        
        # Aspects
        for aspect, keywords in self.aspect_keywords.items():
            score = 0
            count = 0
            for keyword in keywords:
                if keyword in text:
                    count += 1
                    context = self._get_context(text, keyword)
                    score += self._basic_sentiment(context)
            if count > 0:
                review.aspects[aspect] = score / count
        
        # Emotions
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for w in keywords if w in text)
            if score > 0:
                review.emotions[emotion] = min(score / len(keywords), 1.0)
        
        # Credibility and spam
        review.credibility = self._calc_credibility(review)
        review.is_spam = self._detect_spam(review)

    def _get_context(self, text: str, keyword: str, window: int = 10) -> str:
        """Get context around keyword"""
        words = text.split()
        try:
            # Need a robust way to find the index of the keyword, especially
            # if the keyword is a substring of a word. Simple split/index is prone to error.
            # Using find for a simpler fallback, though the whole word approach is better.
            try:
                idx = words.index(keyword)
            except ValueError:
                return text[:100] # Fallback if not found as a separate word

            start = max(0, idx - window)
            end = min(len(words), idx + window + 1)
            return ' '.join(words[start:end])
        except Exception:
            return text[:100]

    def _calc_credibility(self, review: ReviewData) -> float:
        """Calculate credibility score"""
        score = 0.5
        
        # Length factor
        length = len(review.content)
        if 100 <= length <= 500:
            score += 0.2
        
        # Specific details
        specific = ['camera', 'battery', 'screen', 'performance']
        mentions = sum(1 for w in specific if w in review.content.lower())
        score += min(mentions * 0.1, 0.3)
        
        # Balanced sentiment
        if -0.3 <= review.sentiment <= 0.8:
            score += 0.1
        
        # Database bonus
        if review.from_database:
            score += 0.1
        
        return min(score, 1.0)

    def _detect_spam(self, review: ReviewData) -> bool:
        """Detect spam reviews"""
        text = review.content.lower()
        spam_patterns = ['buy now', 'click here', 'deal', 'promo', 'discount']
        spam_score = sum(1 for p in spam_patterns if p in text)
        
        # Unrealistic length
        if len(review.content) < 20 or len(review.content) > 2000:
            spam_score += 1
        
        return spam_score >= 2

    def _perform_analysis(self, phone_name: str, reviews: List[ReviewData]) -> PhoneAnalysis:
        """Perform comprehensive analysis"""
        
        # Filter spam
        clean_reviews = [r for r in reviews if not r.is_spam]
        spam_pct = ((len(reviews) - len(clean_reviews)) / len(reviews) * 100) if reviews else 0
        
        if not clean_reviews:
            return self._empty_analysis(phone_name)
        
        # Rating analysis
        ratings = [r.rating for r in clean_reviews if r.rating]
        avg_rating = np.mean(ratings) if ratings else None
        
        # Sentiment distribution
        sentiments = [r.sentiment for r in clean_reviews]
        pos = sum(1 for s in sentiments if s > 0.1)
        neg = sum(1 for s in sentiments if s < -0.1)
        neu = len(clean_reviews) - pos - neg
        
        sentiment_dist = {
            'positive': pos / len(clean_reviews),
            'negative': neg / len(clean_reviews),
            'neutral': neu / len(clean_reviews)
        }
        
        # Aspect sentiments
        aspect_sents = {}
        for aspect in self.aspect_keywords.keys():
            data = {'positive': 0, 'negative': 0, 'neutral': 0}
            for review in clean_reviews:
                if aspect in review.aspects:
                    sent = review.aspects[aspect]
                    if sent > 0.1:
                        data['positive'] += 1
                    elif sent < -0.1:
                        data['negative'] += 1
                    else:
                        data['neutral'] += 1
            if sum(data.values()) > 0:
                aspect_sents[aspect] = data
        
        # Pros and cons
        pros = self._extract_pros(clean_reviews)
        cons = self._extract_cons(clean_reviews)
        
        # Credibility
        credibility = np.mean([r.credibility for r in clean_reviews])
        
        # Emotions
        emotions = {}
        for emotion in self.emotion_keywords.keys():
            scores = [r.emotions.get(emotion, 0) for r in clean_reviews]
            non_zero = [s for s in scores if s > 0]
            emotions[emotion] = np.mean(non_zero) if non_zero else 0
        
        # Quotes
        quotes = self._extract_quotes(clean_reviews)
        
        # Recommendation score
        rec_score = self._calc_recommendation(sentiment_dist, credibility, aspect_sents)
        
        # Sources
        sources = list(set([r.source for r in clean_reviews]))
        
        return PhoneAnalysis(
            phone_name=phone_name,
            total_reviews=len(clean_reviews),
            avg_rating=avg_rating,
            sentiment_distribution=sentiment_dist,
            aspect_sentiments=aspect_sents,
            top_pros=pros,
            top_cons=cons,
            credibility_score=credibility,
            spam_percentage=spam_pct,
            emotional_profile=emotions,
            user_quotes=quotes,
            recommendation_score=rec_score,
            sources=sources
        )

    def _extract_pros(self, reviews: List[ReviewData]) -> List[str]:
        """Extract pros"""
        positive = [r for r in reviews if r.sentiment > 0.2]
        aspect_counts = defaultdict(int)
        
        for r in positive:
            for aspect, sent in r.aspects.items():
                if sent > 0.3:
                    aspect_counts[aspect] += 1
        
        pros = []
        for aspect, count in sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            pct = (count / len(reviews)) * 100
            pros.append(f"Excellent {aspect} ({pct:.0f}% positive)")
        
        return pros

    def _extract_cons(self, reviews: List[ReviewData]) -> List[str]:
        """Extract cons"""
        negative = [r for r in reviews if r.sentiment < -0.2]
        aspect_counts = defaultdict(int)
        
        for r in negative:
            for aspect, sent in r.aspects.items():
                if sent < -0.3:
                    aspect_counts[aspect] += 1
        
        cons = []
        for aspect, count in sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            pct = (count / len(reviews)) * 100
            cons.append(f"{aspect.capitalize()} issues ({pct:.0f}% negative)")
        
        return cons

    def _extract_quotes(self, reviews: List[ReviewData]) -> List[str]:
        """Extract representative quotes"""
        quotes = []
        
        # Best positive
        positive = [r for r in reviews if r.sentiment > 0.5 and len(r.content) > 50]
        if positive:
            best = max(positive, key=lambda x: x.credibility * x.sentiment)
            quote = best.content[:200] + ("..." if len(best.content) > 200 else "")
            quotes.append(quote)
        
        # Most credible
        credible = sorted(reviews, key=lambda x: x.credibility, reverse=True)
        if credible and (not positive or credible[0] not in positive):
            quote = credible[0].content[:200] + ("..." if len(credible[0].content) > 200 else "")
            quotes.append(quote)
        
        return quotes[:3]

    def _calc_recommendation(self, sentiment_dist: Dict, credibility: float, aspects: Dict) -> float:
        """Calculate recommendation score"""
        base = sentiment_dist['positive'] * 100
        cred_bonus = (credibility - 0.5) * 20
        aspect_bonus = 0
        
        for data in aspects.values():
            total = sum(data.values())
            if total > 0:
                aspect_bonus += (data['positive'] / total - 0.5) * 5
        
        penalty = sentiment_dist['negative'] * 30
        score = base + cred_bonus + aspect_bonus - penalty
        
        return max(0, min(100, score))

    def _empty_analysis(self, phone_name: str) -> PhoneAnalysis:
        """Empty analysis"""
        return PhoneAnalysis(
            phone_name=phone_name,
            total_reviews=0,
            avg_rating=None,
            sentiment_distribution={'positive': 0, 'negative': 0, 'neutral': 0},
            aspect_sentiments={},
            top_pros=[],
            top_cons=[],
            credibility_score=0,
            spam_percentage=0,
            emotional_profile={},
            user_quotes=[],
            recommendation_score=0,
            sources=[]
        )

# CACHE REMOVED: Rely on st.session_state or let it re-initialize
def get_engine():
    """Get the engine instance (not cached due to requests.Session)"""
    if 'hybrid_engine' not in st.session_state:
        st.session_state['hybrid_engine'] = HybridReviewEngine()
    return st.session_state['hybrid_engine']

# ----------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------

def display_sentiment_chart(analysis: PhoneAnalysis):
    """Displays a donut chart of the overall sentiment distribution."""
    if not PLOTLY_AVAILABLE:
        st.warning("Cannot display chart: Plotly not available.")
        return

    data = analysis.sentiment_distribution
    if not any(data.values()):
        st.info("Not enough sentiment data for a chart.")
        return

    df = pd.DataFrame(list(data.items()), columns=['Sentiment', 'Proportion'])
    df['Proportion'] = df['Proportion'] * 100

    fig = go.Figure(data=[go.Pie(
        labels=df['Sentiment'],
        values=df['Proportion'],
        hole=.4,
        marker_colors=['#4CAF50', '#F44336', '#FFEB3B'],
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
    )])

    fig.update_layout(
        title_text=f"Overall Sentiment: {analysis.total_reviews} Reviews",
        title_x=0.5,
        height=300,
        margin=dict(t=50, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def display_aspect_chart(analysis: PhoneAnalysis):
    """Displays a bar chart of sentiment by product aspect."""
    if not PLOTLY_AVAILABLE:
        st.warning("Cannot display chart: Plotly not available.")
        return

    if not analysis.aspect_sentiments:
        st.info("No detailed aspect sentiment data available.")
        return

    aspects = analysis.aspect_sentiments.keys()
    plot_data = []

    for aspect, data in analysis.aspect_sentiments.items():
        total = sum(data.values())
        if total > 0:
            plot_data.append({
                'Aspect': aspect.capitalize(),
                'Positive': data['positive'] / total,
                'Negative': data['negative'] / total,
                'Neutral': data['neutral'] / total
            })

    if not plot_data:
        st.info("No meaningful aspect sentiment data to plot.")
        return
        
    df_plot = pd.DataFrame(plot_data)
    df_plot = df_plot.melt(id_vars=['Aspect'], var_name='Sentiment', value_name='Proportion')

    fig = px.bar(
        df_plot,
        x='Aspect',
        y='Proportion',
        color='Sentiment',
        color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FFEB3B'},
        title="Sentiment by Key Aspect",
        height=350
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Proportion of Mentions",
        legend_title="",
        margin=dict(t=50, b=0, l=0, r=0),
        hovermode="x unified",
        barmode='relative'
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def display_word_cloud(analysis: PhoneAnalysis):
    """Generates and displays a word cloud from the most positive/negative reviews."""
    if not WORDCLOUD_AVAILABLE or not NLTK_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        st.warning("Cannot display word cloud: Missing `wordcloud`, `nltk`, or `matplotlib`.")
        return
        
    all_text = " ".join(analysis.user_quotes) + " " + " ".join(analysis.top_pros) + " ".join(analysis.top_cons)
    
    if not all_text:
        st.info("Not enough text data to generate a word cloud.")
        return
        
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        stop_words = set()
        
    stop_words.update(analysis.phone_name.lower().split())

    try:
        wc = WordCloud(
            background_color="white",
            max_words=100,
            stopwords=stop_words,
            contour_width=3,
            contour_color='steelblue',
            width=800, height=400
        )
        wc.generate(all_text.lower())

        st.subheader("Key Topics Word Cloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Word Cloud generation failed: {e}")


# ----------------------------------------------------------------------
# STREAMLIT UI LAYOUT FUNCTIONS
# ----------------------------------------------------------------------

def display_analysis_summary(analysis: PhoneAnalysis):
    """Displays the key metrics and overall summary."""
    st.markdown(f"<h1 class='main-header'>📱 {analysis.phone_name} Review Analysis</h1>", unsafe_allow_html=True)

    # 1. Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    # Average Rating
    rating_display = f"{analysis.avg_rating:.1f} / 5.0" if analysis.avg_rating is not None else "N/A"
    col1.markdown(f"""
        <div class="metric-container">
            <p style='font-size:1.2rem; color:#667eea; margin-bottom: 0.5rem;'>Avg. Rating</p>
            <p style='font-size:2rem; font-weight:bold; margin: 0;'>{rating_display}</p>
        </div>
    """, unsafe_allow_html=True)

    # Recommendation Score
    col2.markdown(f"""
        <div class="metric-container">
            <p style='font-size:1.2rem; color:#667eea; margin-bottom: 0.5rem;'>Rec. Score</p>
            <p style='font-size:2rem; font-weight:bold; margin: 0; color: {'#4CAF50' if analysis.recommendation_score > 60 else '#F44336' if analysis.recommendation_score < 40 else '#FFEB3B'}'>
                {analysis.recommendation_score:.0f}%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Total Reviews
    col3.markdown(f"""
        <div class="metric-container">
            <p style='font-size:1.2rem; color:#667eea; margin-bottom: 0.5rem;'>Total Reviews</p>
            <p style='font-size:2rem; font-weight:bold; margin: 0;'>{analysis.total_reviews:,}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Credibility Score
    col4.markdown(f"""
        <div class="metric-container">
            <p style='font-size:1.2rem; color:#667eea; margin-bottom: 0.5rem;'>Credibility</p>
            <p style='font-size:2rem; font-weight:bold; margin: 0;'>{analysis.credibility_score*100:.0f}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Charts and Pros/Cons
    col_chart, col_pros_cons = st.columns([2, 1])

    with col_chart:
        # Sentiment Chart
        display_sentiment_chart(analysis)
        
    with col_pros_cons:
        st.subheader("Verdict Highlights")
        
        # Pros
        st.success("👍 Top 3 Pros")
        if analysis.top_pros:
            for pro in analysis.top_pros:
                st.write(f"- {pro}")
        else:
            st.info("No strong positive aspects identified.")
            
        st.markdown("")
        
        # Cons
        st.error("👎 Top 3 Cons")
        if analysis.top_cons:
            for con in analysis.top_cons:
                st.write(f"- {con}")
        else:
            st.info("No strong negative aspects identified.")

    st.markdown("---")

    # 3. Deep Dive
    st.header("Deep Dive Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Aspect Sentiment", "User Voices", "Data Sources", "Emotions & Spam"])
    
    with tab1:
        st.subheader("Performance Breakdown by Feature")
        display_aspect_chart(analysis)
        
        # Additional table for clarity
        if analysis.aspect_sentiments:
            st.markdown("### Aspect Score Table")
            aspect_data = {}
            for aspect, sents in analysis.aspect_sentiments.items():
                total = sum(sents.values())
                aspect_data[aspect.capitalize()] = {
                    'Positive (%)': f"{(sents['positive']/total)*100:.1f}",
                    'Negative (%)': f"{(sents['negative']/total)*100:.1f}",
                    'Total Mentions': total
                }
            st.dataframe(pd.DataFrame(aspect_data).T.sort_values(by='Total Mentions', ascending=False), use_container_width=True)

    with tab2:
        st.subheader("Direct User Feedback")
        if analysis.user_quotes:
            for quote in analysis.user_quotes:
                st.markdown(f"<div class='review-card'>**💬 Quote:** *{quote}*</div>", unsafe_allow_html=True)
        else:
            st.info("No representative quotes extracted.")
            
        if MATPLOTLIB_AVAILABLE:
            display_word_cloud(analysis)

    with tab3:
        st.subheader("Review Data Sources")
        st.metric("Reviews from Database", f"{analysis.db_reviews_count:,}")
        st.metric("Reviews from Web Search", f"{analysis.web_reviews_count:,}")
        
        st.markdown("### Sources Used")
        if analysis.sources:
            for source in analysis.sources:
                st.write(f"- {source}")
        else:
            st.info("Source information not captured or available.")

    with tab4:
        st.subheader("Emotional Profile & Data Quality")
        
        col_em, col_spam = st.columns(2)
        with col_em:
            st.markdown("#### Top User Emotions (Average Score)")
            emotion_df = pd.DataFrame(list(analysis.emotional_profile.items()), columns=['Emotion', 'Score'])
            emotion_df = emotion_df[emotion_df['Score'] > 0].sort_values(by='Score', ascending=False)
            if not emotion_df.empty:
                 st.dataframe(emotion_df, hide_index=True, use_container_width=True)
            else:
                 st.info("No strong emotions detected.")

        with col_spam:
            st.markdown("#### Data Quality Metrics")
            st.metric("Spam/Low-Quality Reviews", f"{analysis.spam_percentage:.1f}%")
            st.metric("Overall Credibility Score", f"{analysis.credibility_score*100:.1f}% (0-100)")
            st.info("A higher Credibility Score indicates more detailed, balanced, and diverse reviews.")


def main_app():
    """Streamlit main application function."""
    
    # Initialize Engine (will use st.session_state)
    engine = get_engine()

    # Sidebar
    st.sidebar.header("Configuration")
    phone_name = st.sidebar.text_input(
        "Enter Phone Model Name",
        "Samsung Galaxy S23 Ultra",
        help="E.g., iPhone 15 Pro, Google Pixel 8, OnePlus 12"
    )
    
    force_web_search = st.sidebar.checkbox(
        "Force Web Search (Ignore Database)",
        False,
        help="Check this to skip the local database and only use web scraping."
    )
    
    st.sidebar.markdown(f"**Database Status:** *{engine.db_manager.load_status}*")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Package Status")
    st.sidebar.markdown(f"Plotly: {'✅ Available' if PLOTLY_AVAILABLE else '❌ Missing'}")
    st.sidebar.markdown(f"WordCloud: {'✅ Available' if WORDCLOUD_AVAILABLE else '❌ Missing'}")
    st.sidebar.markdown(f"NLTK: {'✅ Available' if NLTK_AVAILABLE else '❌ Missing'}")
    st.sidebar.markdown(f"TextBlob: {'✅ Available' if TEXTBLOB_AVAILABLE else '❌ Missing'}")
    
    # Main content
    if phone_name:
        if st.sidebar.button("Analyze Phone Reviews", type="primary"):
            # Clear previous results before a new run
            if 'analysis_result' in st.session_state:
                del st.session_state['analysis_result'] 
            
            st.session_state['analysis_requested'] = True
            st.session_state['phone_name'] = phone_name
            st.session_state['force_web'] = force_web_search
            
        if 'analysis_requested' in st.session_state and st.session_state['analysis_requested']:
            
            # Function to run analysis (NOT CACHED due to serialization errors)
            # The engine itself is reused via get_engine/session_state, and the 
            # heavy data loading is cached via load_database_df.
            def run_analysis(name, force_web):
                return engine.analyze_phone(name, force_web)
            
            # Use st.session_state to hold the result so it persists between reruns 
            # *if* a new analysis isn't requested.
            if 'analysis_result' not in st.session_state:
                st.session_state['analysis_result'] = run_analysis(
                    st.session_state['phone_name'], 
                    st.session_state['force_web']
                )

            analysis = st.session_state['analysis_result']
            
            if analysis.total_reviews > 0:
                display_analysis_summary(analysis)
            else:
                st.error(f"❌ Could not find any reviews for **{st.session_state['phone_name']}** from the database or web search. Try a different model name.")

    else:
        st.markdown(f"<h1 class='main-header'>AI Phone Review Engine</h1>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="feature-card">
        ### Welcome to the Hybrid Review Engine
        This app aggregates and analyzes phone reviews from a local database and a live web search.
        
        **How it works:**
        1. **Database Check:** First, it searches the local `final_dataset_streamlined_clean.csv`.
        2. **Web Scrape:** If the review count is below **{engine.min_reviews_threshold}**, it searches the web for more reviews.
        3. **AI Analysis:** Sentiment, aspect, emotion, and credibility analysis is performed on all collected reviews.
        
        **👈 Start by entering a phone model name in the sidebar.**
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    # Initialize session state for the engine before main_app runs
    if 'hybrid_engine' not in st.session_state:
        get_engine()
        
    main_app()
