


import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple, List, Dict, Any
from nltk.tokenize import sent_tokenize
import argparse

# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


# --- New Functions for Alternative Search Engines ---

def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a search using DuckDuckGo's HTML API.
    
    Args:
        query (str): Search query.
        max_results (int): Maximum number of results to return.
        
    Returns:
        List[Dict[str, Any]]: List of search results.
    """
    try:
        # DuckDuckGo search URL
        search_url = f"https://html.duckduckgo.com/html/"
        params = {
            'q': query,
            's': '0',  # Start from the first result
            'dc': '20',  # Request more results than needed to ensure we get enough
            'kl': 'us-en',  # Region and language
        }
        
        print(f"Performing DuckDuckGo search for: {query}")
        response = session.post(search_url, data=params, timeout=20)
        response.raise_for_status()
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find all result elements
        result_elements = soup.select('.result')
        
        for i, element in enumerate(result_elements):
            if i >= max_results:
                break
                
            # Extract title and URL
            title_element = element.select_one('.result__title')
            url_element = element.select_one('.result__url')
            snippet_element = element.select_one('.result__snippet')
            
            if title_element and url_element:
                title = title_element.get_text(strip=True)
                url = url_element.get('href', '')
                
                # Clean up the URL (DuckDuckGo uses redirects)
                if '/uddg=' in url:
                    url = url.split('/uddg=')[1].split('&')[0]
                    url = requests.utils.unquote(url)
                
                # Extract snippet if available
                snippet = ""
                if snippet_element:
                    snippet = snippet_element.get_text(strip=True)
                
                # Extract site name from URL
                site_name = url.split('/')[2] if '://' in url else url.split('/')[0]
                
                result = {
                    'id': i + 1,
                    'title': title,
                    'url': url,
                    'site_name': site_name,
                    'date': '',  # DuckDuckGo doesn't provide dates directly
                    'snippet': snippet,
                    'context': ''  # Reserved field to be filled later
                }
                results.append(result)
        
        print(f"Found {len(results)} results from DuckDuckGo")
        return results
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        return []


def jina_search(query: str, jina_api_key: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a search using Jina's API.
    
    Args:
        query (str): Search query.
        jina_api_key (str): API key for Jina.
        max_results (int): Maximum number of results to return.
        
    Returns:
        List[Dict[str, Any]]: List of search results.
    """
    try:
        # Define Jina search endpoint
        search_url = "https://api.jina.ai/v1/search"
        
        headers = {
            'Authorization': f'Bearer {jina_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "query": query,
            "top_k": max_results
        }
        
        print(f"Performing Jina search for: {query}")
        response = requests.post(search_url, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        
        search_results = response.json()
        results = []
        
        # Process results
        if 'results' in search_results:
            for i, result in enumerate(search_results['results']):
                if i >= max_results:
                    break
                    
                # Extract information from result
                url = result.get('url', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                
                # Extract domain from URL as site name
                site_name = url.split('/')[2] if '://' in url else url.split('/')[0]
                
                result_data = {
                    'id': i + 1,
                    'title': title,
                    'url': url,
                    'site_name': site_name,
                    'date': result.get('date', ''),
                    'snippet': snippet,
                    'context': ''  # Reserved field to be filled later
                }
                results.append(result_data)
        
        print(f"Found {len(results)} results from Jina")
        return results
    except Exception as e:
        print(f"Error in Jina search: {str(e)}")
        return []


def bing_web_search(query: str, subscription_key: str, endpoint: str, market: str = 'en-US', language: str = 'en', max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a search using Bing Search API.
    
    Args:
        query (str): Search query.
        subscription_key (str): Azure Cognitive Services subscription key.
        endpoint (str): Bing Search API endpoint.
        market (str): Market code (e.g., 'en-US').
        language (str): Language code (e.g., 'en').
        max_results (int): Maximum number of results to return.
        
    Returns:
        List[Dict[str, Any]]: List of search results.
    """
    try:
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key
        }
        params = {
            'q': query,
            'mkt': market,
            'setLang': language,
            'count': max_results,
            'responseFilter': 'Webpages'
        }
        
        response = session.get(endpoint, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        search_results = response.json()
        
        results = []
        if 'webPages' in search_results and 'value' in search_results['webPages']:
            for i, item in enumerate(search_results['webPages']['value']):
                result = {
                    'id': i + 1,
                    'title': item.get('name', ''),
                    'url': item.get('url', ''),
                    'site_name': item.get('displayUrl', '').split('/')[2] if '//' in item.get('displayUrl', '') else '',
                    'date': '',  # Bing does not provide publication date in basic results
                    'snippet': item.get('snippet', ''),
                    'context': ''
                }
                results.append(result)
        
        print(f"Found {len(results)} results from Bing")
        return results
    except Exception as e:
        print(f"Error in Bing search: {str(e)}")
        return []

def extract_relevant_info(search_results: List[Dict[str, Any]], max_results: int = 10) -> List[Dict[str, str]]:
    """
    Extract relevant information from search results (works with DuckDuckGo/Jina/Bing formats).
    
    Args:
        search_results: Raw search results from any supported engine
        max_results: Maximum number of results to return
        
    Returns:
        List of processed results with standardized format
    """
    processed_results = []
    for result in search_results[:max_results]:
        processed = {
            'title': result.get('title', 'No title available'),
            'url': result.get('url', ''),
            'snippet': result.get('snippet', 'No snippet available'),
            'site_name': result.get('site_name', ''),
            'date': result.get('date', ''),
            'context': result.get('context', '')
        }
        # Clean special characters from snippet
        processed['snippet'] = re.sub(r'[\xa0\n\t]+', ' ', processed['snippet']).strip()
        processed_results.append(processed)
    return processed_results

def main():
    parser = argparse.ArgumentParser(description='Run search with various search engines')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--split', type=str, required=True, help='Dataset split')
    parser.add_argument('--max_search_limit', type=int, default=5, help='Maximum search limit')
    parser.add_argument('--max_turn', type=int, default=10, help='Maximum number of turns')
    parser.add_argument('--top_k', type=int, default=10, help='Top k results to return')
    parser.add_argument('--max_doc_len', type=int, default=3000, help='Maximum document length')
    parser.add_argument('--use_jina', type=str, default='False', choices=['True', 'False'], help='Whether to use Jina for content extraction')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--jina_api_key', type=str, help='API key for Jina')
    parser.add_argument('--search_engine', type=str, default='duckduckgo', choices=['bing', 'duckduckgo', 'jina'], help='Search engine to use')
    parser.add_argument('--bing_subscription_key', type=str, help='Subscription key for Bing Search API (only needed if using Bing)')
    
    args = parser.parse_args()
    
    # Convert string 'True'/'False' to boolean
    use_jina_extraction = args.use_jina.lower() == 'true'
    
    # Check if required API keys are provided based on the search engine
    if args.search_engine == 'bing' and not args.bing_subscription_key:
        parser.error("--bing_subscription_key is required when using Bing search engine")
    
    if args.search_engine == 'jina' and not args.jina_api_key:
        parser.error("--jina_api_key is required when using Jina search engine")
    
    if use_jina_extraction and not args.jina_api_key:
        parser.error("--jina_api_key is required when using Jina for content extraction (--use_jina True)")
    
    # Example query - in a real scenario, this would come from your dataset
    query = f"Example query from {args.dataset_name} {args.split} dataset"