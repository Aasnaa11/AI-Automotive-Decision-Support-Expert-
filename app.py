"""
AI Automotive Decision Support Expert - Powered by Llama 3.2 via Ollama
==========================================================================
A production-ready, domain-specific AI system with hybrid architecture for automotive consulting.

Author: Advanced AI Coursework Submission
Version: 1.0
Framework: Streamlit + Ollama (Llama 3.2)
Architecture: Hybrid (Rule-Based Classification + LLM Response Engine)
"""

import streamlit as st
import requests
import json
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 800

# =============================================================================
# RULE-BASED CLASSIFICATION ENGINE (LAYER 1)
# =============================================================================

class AutomotiveQueryClassifier:
    """
    Intelligent rule-based classification engine that categorizes user queries
    into specific automotive domains before LLM processing.
    """
    
    # Classification categories with associated keywords
    CATEGORIES = {
        "Car Specifications": [
            "specs", "specification", "horsepower", "hp", "torque", "engine size",
            "displacement", "cylinders", "dimensions", "wheelbase", "ground clearance",
            "boot space", "trunk", "seating capacity", "fuel tank", "0-60", "0-100"
        ],
        "Car Comparison": [
            "vs", "versus", "compare", "comparison", "better than", "difference between",
            "which is better", "choose between", "or", "against"
        ],
        "Engine/Mechanical Explanation": [
            "how does", "how work", "explain", "what is", "turbo", "supercharger",
            "cvt", "dct", "transmission", "differential", "suspension", "brakes",
            "abs", "ebd", "traction control", "camshaft", "crankshaft", "piston"
        ],
        "Maintenance Advice": [
            "service", "maintenance", "oil change", "when to replace", "tire rotation",
            "brake pad", "air filter", "spark plug", "coolant", "transmission fluid",
            "schedule", "interval", "cost of maintenance"
        ],
        "Electric Vehicles": [
            "electric", "ev", "battery", "charging", "range", "bev", "phev",
            "hybrid", "kwh", "charging time", "charging station", "regenerative",
            "electric motor", "tesla", "leaf", "id.4"
        ],
        "Buying Recommendation": [
            "should i buy", "recommend", "suggestion", "best car", "good car",
            "worth buying", "purchase", "looking for", "want to buy", "advice on buying"
        ],
        "Motorsport": [
            "f1", "formula 1", "racing", "nascar", "rally", "wrc", "le mans",
            "motorsport", "race car", "track", "circuit", "lap time", "drag race"
        ],
        "Troubleshooting": [
            "problem", "issue", "noise", "sound", "vibration", "warning light",
            "check engine", "smoke", "leak", "won't start", "stalling", "overheating",
            "rough idle", "squeaking", "grinding"
        ],
        "Budget-Based Recommendation": [
            "budget", "price range", "afford", "under", "within", "family car",
            "economical", "fuel efficient", "cheap", "value for money", "cost effective"
        ]
    }
    
    @staticmethod
    def classify_query(query: str) -> str:
        """
        Classify user query into one of the predefined categories using keyword matching.
        
        Args:
            query: User input string
            
        Returns:
            Category name as string
        """
        query_lower = query.lower()
        category_scores = {}
        
        # Calculate relevance score for each category
        for category, keywords in AutomotiveQueryClassifier.CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            category_scores[category] = score
        
        # Return category with highest score, default to "Car Specifications"
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        
        return "Car Specifications"  # Default fallback
    
    @staticmethod
    def get_category_prompt_enhancement(category: str) -> str:
        """
        Generate category-specific prompt enhancements to guide LLM responses.
        
        Args:
            category: Classified category name
            
        Returns:
            Enhanced prompt instruction string
        """
        enhancements = {
            "Car Specifications": """
                Provide detailed technical specifications in a structured format.
                Include: Engine, Power/Torque, Dimensions, Features, Performance metrics.
                Use bullet points and organize by category.
            """,
            "Car Comparison": """
                Create a comprehensive comparison table covering:
                - Engine & Performance
                - Fuel Efficiency
                - Safety Features
                - Technology & Infotainment
                - Price & Value
                - Maintenance Costs
                Conclude with a clear recommendation based on different use cases.
            """,
            "Engine/Mechanical Explanation": """
                Explain the technical concept clearly using:
                1. Simple definition
                2. How it works (step-by-step)
                3. Benefits and drawbacks
                4. Real-world examples
                Use analogies for complex concepts.
            """,
            "Maintenance Advice": """
                Provide practical maintenance guidance:
                - Recommended intervals (time/mileage)
                - Expected costs (parts + labor estimates)
                - DIY vs Professional advice
                - Warning signs to watch for
                Include disclaimer about consulting professional mechanics.
            """,
            "Electric Vehicles": """
                Focus on EV-specific aspects:
                - Battery technology and capacity
                - Real-world range analysis
                - Charging infrastructure and times
                - Total cost of ownership vs ICE
                - Environmental impact
                Provide data-backed comparisons.
            """,
            "Buying Recommendation": """
                Provide structured buying advice:
                - Vehicle suitability analysis
                - Pros and cons
                - Ownership cost projection (3-5 years)
                - Resale value outlook
                - Alternative options
                Ask clarifying questions if needed (budget, usage, preferences).
            """,
            "Motorsport": """
                Provide expert motorsport insights:
                - Technical regulations and specifications
                - Performance analysis
                - Historical context
                - Driver/team comparisons
                Use enthusiast-level technical detail.
            """,
            "Troubleshooting": """
                Provide diagnostic guidance:
                - Possible causes (most likely to least likely)
                - Basic diagnostic steps
                - Safety warnings
                - When to seek professional help
                - Estimated repair costs
                **IMPORTANT**: Always include disclaimer to consult certified mechanics.
            """,
            "Budget-Based Recommendation": """
                Provide comprehensive budget-based analysis:
                - 3 vehicle recommendations at different price points
                - Feature comparison
                - Ownership cost breakdown (fuel, insurance, maintenance)
                - Suitability score for stated requirements
                - Trade-offs and compromises
                Structure as a decision matrix.
            """
        }
        
        return enhancements.get(category, "Provide a clear and structured response.")

# =============================================================================
# LANGUAGE DETECTION & SUPPORT
# =============================================================================

class LanguageDetector:
    """
    Simple heuristic-based language detection system supporting 10 languages.
    """
    
    LANGUAGE_PATTERNS = {
        "Arabic": re.compile(r'[\u0600-\u06FF]'),
        "Hindi": re.compile(r'[\u0900-\u097F]'),
        "Urdu": re.compile(r'[\u0600-\u06FF\u0750-\u077F]'),
        "Chinese": re.compile(r'[\u4E00-\u9FFF]'),
        "Japanese": re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
        "Turkish": re.compile(r'\b(ı|ğ|ü|ş|ö|ç)\b', re.IGNORECASE),
        "French": re.compile(r'\b(le|la|les|un|une|des|est|sont|avec|pour)\b', re.IGNORECASE),
        "Spanish": re.compile(r'\b(el|la|los|las|un|una|es|son|con|para|¿|¡)\b', re.IGNORECASE),
        "German": re.compile(r'\b(der|die|das|und|ist|sind|mit|für|ä|ö|ü|ß)\b', re.IGNORECASE),
    }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect language using character patterns and common words.
        
        Args:
            text: Input text string
            
        Returns:
            Detected language name
        """
        for language, pattern in LanguageDetector.LANGUAGE_PATTERNS.items():
            if pattern.search(text):
                return language
        
        return "English"  # Default fallback
    
    @staticmethod
    def get_multilingual_instruction(language: str) -> str:
        """
        Generate instruction for LLM to respond in detected language.
        
        Args:
            language: Detected language name
            
        Returns:
            Instruction string
        """
        if language == "English":
            return ""
        
        return f"\n**IMPORTANT**: Respond in {language} language to match the user's input."

# =============================================================================
# OLLAMA LLM INTERFACE (LAYER 2)
# =============================================================================

class OllamaLLMEngine:
    """
    Interface to Ollama API for Llama 3.2 model with streaming support.
    """
    
    @staticmethod
    def generate_response(
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = True
    ) -> Optional[str]:
        """
        Generate response from Ollama API with streaming support.
        
        Args:
            prompt: Complete prompt including system and user messages
            temperature: Sampling temperature (0.1 - 1.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Enable streaming response
            
        Returns:
            Generated response text or None on error
        """
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                stream=stream,
                timeout=60
            )
            response.raise_for_status()
            
            if stream:
                return OllamaLLMEngine._handle_streaming_response(response)
            else:
                return response.json().get("response", "")
                
        except requests.exceptions.ConnectionError:
            return "❌ **Connection Error**: Cannot connect to Ollama. Please ensure Ollama is running (`ollama serve`) and the model is available (`ollama run llama3.2`)."
        except requests.exceptions.Timeout:
            return "❌ **Timeout Error**: The request took too long. Please try again."
        except requests.exceptions.RequestException as e:
            return f"❌ **Error**: {str(e)}"
        except Exception as e:
            return f"❌ **Unexpected Error**: {str(e)}"
    
    @staticmethod
    def _handle_streaming_response(response) -> str:
        """
        Handle streaming response from Ollama API.
        
        Args:
            response: Requests response object with streaming enabled
            
        Returns:
            Complete generated text
        """
        full_response = ""
        
        try:
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'response' in json_response:
                        full_response += json_response['response']
                    if json_response.get('done', False):
                        break
        except json.JSONDecodeError as e:
            return full_response + f"\n\n⚠️ Warning: Partial response due to parsing error."
        
        return full_response

# =============================================================================
# CONVERSATIONAL MEMORY MANAGER
# =============================================================================

class ConversationMemory:
    """
    Manages conversation history with context window management.
    """
    
    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session state for conversation memory."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata dictionary
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            message["metadata"] = metadata
        
        st.session_state.messages.append(message)
        st.session_state.conversation_count += 1
    
    @staticmethod
    def get_recent_context(n: int = 6) -> List[Dict]:
        """
        Retrieve last n conversation turns for context.
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of recent messages
        """
        return st.session_state.messages[-n:] if len(st.session_state.messages) >= n else st.session_state.messages
    
    @staticmethod
    def format_context_for_prompt(context: List[Dict]) -> str:
        """
        Format conversation context for LLM prompt.
        
        Args:
            context: List of message dictionaries
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
        
        formatted = "\n\n**Previous Conversation Context:**\n"
        for msg in context:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"\n{role_label}: {msg['content'][:200]}..."  # Truncate long messages
        
        return formatted + "\n\n---\n"
    
    @staticmethod
    def clear_history():
        """Clear all conversation history."""
        st.session_state.messages = []
        st.session_state.conversation_count = 0

# =============================================================================
# ADVANCED FEATURE: CAR COMPARISON TOOL
# =============================================================================

class CarComparisonEngine:
    """
    Structured car comparison tool with AI-powered analysis.
    """
    
    @staticmethod
    def generate_comparison(car1: str, car2: str, temperature: float) -> str:
        """
        Generate detailed comparison between two vehicles.
        
        Args:
            car1: First car name
            car2: Second car name
            temperature: LLM temperature setting
            
        Returns:
            Formatted comparison analysis
        """
        comparison_prompt = f"""
**ROLE**: You are a senior automotive analyst with expertise in vehicle evaluation.

**TASK**: Create a comprehensive comparison between {car1} and {car2}.

**OUTPUT FORMAT** (MANDATORY):

# {car1} vs {car2} - Detailed Comparison

## 1. Engine & Performance
- **{car1}**: [Engine specs, power, torque, acceleration]
- **{car2}**: [Engine specs, power, torque, acceleration]
- **Winner**: [Car name and reason]

## 2. Fuel Efficiency
- **{car1}**: [City/Highway MPG or L/100km]
- **{car2}**: [City/Highway MPG or L/100km]
- **Winner**: [Car name and reason]

## 3. Safety Features
- **{car1}**: [Safety rating, key features]
- **{car2}**: [Safety rating, key features]
- **Winner**: [Car name and reason]

## 4. Technology & Features
- **{car1}**: [Infotainment, driver aids]
- **{car2}**: [Infotainment, driver aids]
- **Winner**: [Car name and reason]

## 5. Estimated Maintenance Cost (5 years)
- **{car1}**: [Annual cost estimate]
- **{car2}**: [Annual cost estimate]
- **Winner**: [Car name and reason]

## 6. Resale Value Outlook
- **{car1}**: [Depreciation analysis]
- **{car2}**: [Depreciation analysis]
- **Winner**: [Car name and reason]

## 7. Final Recommendation Verdict

**Best for Family Use**: [Car name and reason]
**Best for Fuel Economy**: [Car name and reason]
**Best for Performance**: [Car name and reason]
**Best Overall Value**: [Car name and reason]

**Conclusion**: [2-3 sentence final recommendation]

Use actual data where possible. Be specific and factual.
"""
        
        return OllamaLLMEngine.generate_response(comparison_prompt, temperature=temperature)

# =============================================================================
# ADVANCED FEATURE: BUDGET-BASED RECOMMENDATION ENGINE
# =============================================================================

class BudgetRecommendationEngine:
    """
    AI-powered decision support system for budget-based vehicle recommendations.
    """
    
    @staticmethod
    def generate_recommendation(
        budget_min: int,
        budget_max: int,
        fuel_type: str,
        family_size: int,
        body_type: str,
        usage_city: int,
        usage_highway: int,
        temperature: float
    ) -> str:
        """
        Generate comprehensive budget-based vehicle recommendations.
        
        Args:
            budget_min: Minimum budget in currency
            budget_max: Maximum budget in currency
            fuel_type: Preferred fuel type
            family_size: Number of family members
            body_type: Preferred body type
            usage_city: City driving percentage
            usage_highway: Highway driving percentage
            temperature: LLM temperature setting
            
        Returns:
            Formatted recommendation analysis
        """
        recommendation_prompt = f"""
**ROLE**: You are an expert automotive consultant specializing in vehicle purchasing decisions.

**USER REQUIREMENTS**:
- Budget Range: ${budget_min:,} - ${budget_max:,}
- Fuel Type Preference: {fuel_type}
- Family Size: {family_size} members
- Body Type Preference: {body_type}
- Usage Pattern: {usage_city}% City / {usage_highway}% Highway

**TASK**: Provide 3 vehicle recommendations at different price points within the budget.

**OUTPUT FORMAT** (MANDATORY):

# Budget-Based Vehicle Recommendations

## Recommendation 1: Budget-Friendly Option

**Vehicle**: [Model Name and Year]
**Price**: [Estimated price]
**Suitability Score**: [X/10]

**Key Features**:
- Engine: [Specs]
- Fuel Efficiency: [City/Highway]
- Seating: [Capacity]
- Cargo Space: [Volume]

**Ownership Cost Analysis (5 years)**:
- Fuel: [Annual estimate]
- Insurance: [Annual estimate]
- Maintenance: [Annual estimate]
- **Total Annual Cost**: [Amount]

**Pros**:
- [3-4 key advantages]

**Cons**:
- [2-3 limitations]

---

## Recommendation 2: Mid-Range Option

[Same format as above]

---

## Recommendation 3: Premium Option (Max Budget)

[Same format as above]

---

## Decision Matrix

| Factor | Option 1 | Option 2 | Option 3 |
|--------|----------|----------|----------|
| Initial Cost | $ | $ | $ |
| Fuel Efficiency | rating | rating | rating |
| Safety Rating | rating | rating | rating |
| Features | Basic | Standard | Premium |
| Resale Value | rating | rating | rating |

## Trade-offs Analysis

**If you prioritize fuel economy**: [Recommendation and reasoning]
**If you prioritize space and comfort**: [Recommendation and reasoning]
**If you prioritize features and tech**: [Recommendation and reasoning]

## Final Recommendation

[2-3 sentences with clear guidance based on the user's specific requirements]

**Important Notes**:
- Prices are estimates and vary by location and dealer
- Test drive all options before deciding
- Consider certified pre-owned for better value

Be specific with actual vehicle models. Use realistic pricing and data.
"""
        
        return OllamaLLMEngine.generate_response(recommendation_prompt, temperature=temperature)

# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================

class SystemPromptBuilder:
    """
    Constructs comprehensive system prompts with category-specific enhancements.
    """
    
    BASE_SYSTEM_PROMPT = """
You are a senior automotive engineer and vehicle consultant with 15 years of industry experience in:
- Powertrain systems and engine technology
- Electric vehicle (EV) technology and battery systems
- Vehicle safety analysis and crash test evaluation
- Market analysis and vehicle purchasing consultation
- Motorsport engineering and performance optimization

**YOUR EXPERTISE INCLUDES**:
- Technical specifications and engineering principles
- Comparative vehicle analysis
- Maintenance scheduling and cost estimation
- Fuel efficiency optimization
- Safety feature evaluation
- Resale value projection

**RESPONSE GUIDELINES**:
1. Provide structured, technically accurate, professional responses
2. Use headings, bullet points, and comparison tables where relevant
3. Support claims with data and technical reasoning
4. Maintain clarity while preserving technical authority
5. Acknowledge limitations when specific data is unavailable
6. Use automotive industry terminology appropriately

**SAFETY PROTOCOLS**:
- Refuse guidance on dangerous illegal modifications
- Provide disclaimers before technical repair instructions
- Never provide medical or legal advice
- Avoid unsafe performance tuning recommendations

**FORMATTING STANDARDS**:
- Use markdown for structure (headers, lists, tables, bold)
- Organize complex information hierarchically
- Include numbered steps for procedures
- Use comparison tables for multi-option analysis
"""
    
    @staticmethod
    def build_prompt(
        user_query: str,
        category: str,
        conversation_context: List[Dict],
        detected_language: str,
        temperature: float
    ) -> str:
        """
        Build complete prompt with all enhancements.
        
        Args:
            user_query: User's input query
            category: Classified category
            conversation_context: Recent conversation history
            detected_language: Detected language
            temperature: Current temperature setting
            
        Returns:
            Complete formatted prompt
        """
        # Get category-specific enhancement
        category_enhancement = AutomotiveQueryClassifier.get_category_prompt_enhancement(category)
        
        # Get language instruction
        language_instruction = LanguageDetector.get_multilingual_instruction(detected_language)
        
        # Format conversation context
        context_str = ConversationMemory.format_context_for_prompt(conversation_context)
        
        # Build complete prompt
        complete_prompt = f"""
{SystemPromptBuilder.BASE_SYSTEM_PROMPT}

**QUERY CATEGORY**: {category}

**CATEGORY-SPECIFIC INSTRUCTIONS**:
{category_enhancement}
{language_instruction}

{context_str}

**CURRENT USER QUERY**:
{user_query}

**YOUR RESPONSE** (follow all formatting and content guidelines):
"""
        
        return complete_prompt

# =============================================================================
# SAFETY & CONTENT FILTERING
# =============================================================================

class SafetyFilter:
    """
    Content safety filter for dangerous or inappropriate queries.
    """
    
    DANGEROUS_KEYWORDS = [
        "remove catalytic converter", "disable airbag", "tamper with odometer",
        "illegal modification", "street racing", "disable safety", "bypass emission",
        "drunk driving", "evade", "fraud"
    ]
    
    @staticmethod
    def check_query_safety(query: str) -> Tuple[bool, str]:
        """
        Check if query contains dangerous or inappropriate requests.
        
        Args:
            query: User input query
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        query_lower = query.lower()
        
        for keyword in SafetyFilter.DANGEROUS_KEYWORDS:
            if keyword in query_lower:
                warning = f"""
⚠️ **Safety Notice**: Your query involves potentially dangerous or illegal modifications.

I cannot provide guidance on:
- Removing or disabling safety equipment
- Illegal vehicle modifications
- Emission system tampering
- Activities that endanger yourself or others

**Instead, I can help with**:
- Legal performance upgrades
- Proper maintenance procedures
- Safety feature understanding
- Regulatory compliance

Please rephrase your question or ask about safe alternatives.
"""
                return False, warning
        
        return True, ""

# =============================================================================
# MAIN STREAMLIT APPLICATION
# =============================================================================

def initialize_streamlit_config():
    """Configure Streamlit page settings and styling."""
    st.set_page_config(
        page_title="AI Automotive Expert",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .category-badge {
            background-color: #e8f4f8;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.9rem;
            color: #1f77b4;
            display: inline-block;
            margin: 0.5rem 0;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar() -> Dict:
    """
    Render sidebar with controls and settings.
    
    Returns:
        Dictionary of user settings
    """
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        # Model information
        st.info(f"🤖 **Model**: Llama 3.2 via Ollama")
        
        # Temperature control
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="Controls response creativity. Lower = more focused, Higher = more creative"
        )
        
        # Language indicator
        if 'detected_language' in st.session_state:
            st.success(f"🌍 **Language**: {st.session_state.detected_language}")
        
        st.markdown("---")
        
        # Advanced Features
        st.markdown("### 🔧 Advanced Features")
        
        comparator_mode = st.toggle(
            "🔄 Car Comparison Mode",
            help="Enable structured car comparison tool"
        )
        
        budget_mode = st.toggle(
            "💰 Budget Recommendation Mode",
            help="Enable budget-based vehicle recommendation engine"
        )
        
        st.markdown("---")
        
        # Conversation controls
        st.markdown("### 💬 Conversation")
        
        message_count = len(st.session_state.messages) // 2
        st.metric("Messages", message_count)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            ConversationMemory.clear_history()
            st.rerun()
        
        st.markdown("---")
        
        # Information
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            **AI Automotive Decision Support Expert**
            
            This is a production-ready, domain-specific AI system featuring:
            
            - **Hybrid Architecture**: Rule-based classification + LLM reasoning
            - **Conversational Memory**: Context-aware responses
            - **Advanced Features**: Car comparison & budget recommendations
            - **Multilingual Support**: 10 languages
            - **Safety Filtering**: Content safety checks
            
            **Powered by**: Llama 3.2 via Ollama
            """)
        
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            **Prerequisites**:
            ```bash
            # Install Ollama
            curl -fsSL https://ollama.com/install.sh | sh
            
            # Pull Llama 3.2 model
            ollama run llama3.2
            
            # Install dependencies
            pip install streamlit requests
            ```
            
            **Run Application**:
            ```bash
            streamlit run app.py
            ```
            """)
    
    return {
        "temperature": temperature,
        "comparator_mode": comparator_mode,
        "budget_mode": budget_mode
    }

def render_comparison_form(temperature: float):
    """
    Render car comparison input form.
    
    Args:
        temperature: Current temperature setting
    """
    st.markdown("### 🔄 Car Comparison Tool")
    st.markdown("Compare two vehicles side-by-side with detailed analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        car1 = st.text_input(
            "First Car",
            placeholder="e.g., Toyota Camry 2024",
            help="Enter the full name and year of the first vehicle"
        )
    
    with col2:
        car2 = st.text_input(
            "Second Car",
            placeholder="e.g., Honda Accord 2024",
            help="Enter the full name and year of the second vehicle"
        )
    
    if st.button("⚡ Generate Comparison", type="primary", use_container_width=True):
        if car1.strip() and car2.strip():
            with st.spinner(f"🔍 Analyzing {car1} vs {car2}..."):
                comparison_result = CarComparisonEngine.generate_comparison(
                    car1.strip(),
                    car2.strip(),
                    temperature
                )
                
                # Add to conversation history
                ConversationMemory.add_message(
                    "user",
                    f"Compare {car1} and {car2}",
                    {"feature": "comparison", "car1": car1, "car2": car2}
                )
                ConversationMemory.add_message("assistant", comparison_result)
                
                st.rerun()
        else:
            st.error("⚠️ Please enter both car names to compare")

def render_budget_recommendation_form(temperature: float):
    """
    Render budget-based recommendation input form.
    
    Args:
        temperature: Current temperature setting
    """
    st.markdown("### 💰 Budget-Based Vehicle Recommendation")
    st.markdown("Get personalized vehicle recommendations based on your requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget_min = st.number_input(
            "Minimum Budget ($)",
            min_value=5000,
            max_value=200000,
            value=20000,
            step=1000
        )
        
        fuel_type = st.selectbox(
            "Fuel Type Preference",
            ["Petrol", "Diesel", "Hybrid", "Electric (EV)", "Any"]
        )
        
        body_type = st.selectbox(
            "Body Type Preference",
            ["SUV", "Sedan", "Hatchback", "Crossover", "Truck", "Van", "Any"]
        )
    
    with col2:
        budget_max = st.number_input(
            "Maximum Budget ($)",
            min_value=5000,
            max_value=200000,
            value=35000,
            step=1000
        )
        
        family_size = st.number_input(
            "Family Size (members)",
            min_value=1,
            max_value=10,
            value=4,
            step=1
        )
        
        col_city, col_highway = st.columns(2)
        with col_city:
            usage_city = st.slider("City %", 0, 100, 60)
        with col_highway:
            usage_highway = 100 - usage_city
            st.metric("Highway %", usage_highway)
    
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        if budget_min >= budget_max:
            st.error("⚠️ Maximum budget must be greater than minimum budget")
        else:
            with st.spinner("🔍 Analyzing market options and generating recommendations..."):
                recommendation_result = BudgetRecommendationEngine.generate_recommendation(
                    budget_min,
                    budget_max,
                    fuel_type,
                    family_size,
                    body_type,
                    usage_city,
                    usage_highway,
                    temperature
                )
                
                # Add to conversation history
                query_summary = f"Budget: ${budget_min:,}-${budget_max:,}, {fuel_type}, {body_type}, {family_size} members, {usage_city}% city"
                ConversationMemory.add_message(
                    "user",
                    query_summary,
                    {"feature": "budget_recommendation"}
                )
                ConversationMemory.add_message("assistant", recommendation_result)
                
                st.rerun()

def render_chat_interface(settings: Dict):
    """
    Render main chat interface with message history.
    
    Args:
        settings: Dictionary of user settings from sidebar
    """
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata badges if present
            if "metadata" in message:
                metadata = message["metadata"]
                if "category" in metadata:
                    st.markdown(
                        f'<span class="category-badge">📁 {metadata["category"]}</span>',
                        unsafe_allow_html=True
                    )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about cars, maintenance, buying advice, or comparisons..."):
        # Safety check
        is_safe, warning = SafetyFilter.check_query_safety(prompt)
        
        if not is_safe:
            with st.chat_message("assistant"):
                st.warning(warning)
            return
        
        # Detect language
        detected_language = LanguageDetector.detect_language(prompt)
        st.session_state.detected_language = detected_language
        
        # Classify query
        category = AutomotiveQueryClassifier.classify_query(prompt)
        
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to conversation history
        ConversationMemory.add_message(
            "user",
            prompt,
            {"category": category, "language": detected_language}
        )
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Get conversation context
                context = ConversationMemory.get_recent_context(n=6)
                
                # Build prompt
                complete_prompt = SystemPromptBuilder.build_prompt(
                    prompt,
                    category,
                    context[:-1],  # Exclude current message
                    detected_language,
                    settings["temperature"]
                )
                
                # Generate response
                response = OllamaLLMEngine.generate_response(
                    complete_prompt,
                    temperature=settings["temperature"]
                )
                
                # Display response
                st.markdown(response)
                
                # Show category badge
                st.markdown(
                    f'<span class="category-badge">📁 {category}</span>',
                    unsafe_allow_html=True
                )
                
                # Add to conversation history
                ConversationMemory.add_message(
                    "assistant",
                    response,
                    {"category": category}
                )

def main():
    """Main application entry point."""
    # Initialize configuration
    initialize_streamlit_config()
    
    # Initialize session state
    ConversationMemory.initialize_session_state()
    
    # Render header
    st.markdown('<div class="main-header">🚗 AI Automotive Decision Support Expert</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Llama 3.2 via Ollama | Hybrid AI Architecture</div>', unsafe_allow_html=True)
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main content area
    if settings["comparator_mode"]:
        # Show comparison tool
        render_comparison_form(settings["temperature"])
        st.markdown("---")
    
    if settings["budget_mode"]:
        # Show budget recommendation tool
        render_budget_recommendation_form(settings["temperature"])
        st.markdown("---")
    
    # Always show chat interface
    if not settings["comparator_mode"] and not settings["budget_mode"]:
        st.markdown("### 💬 Chat with Automotive Expert")
    
    render_chat_interface(settings)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🎓 Advanced AI Coursework Submission | Built with Streamlit + Ollama (Llama 3.2)<br>
        ⚠️ <b>Disclaimer</b>: Always consult certified professionals for mechanical repairs and modifications
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
