"""
Comprehensive Demo: Using Almost Every Feature of Parsec

This example demonstrates:
1. OpenAI adapter with structured output
2. Complex Pydantic validation with deeply nested models
3. Caching with statistics
4. Prompt templates with versioning
5. Template registry and persistence
6. Template manager
7. Dataset collection for training
8. Template analytics and A/B testing
9. Automatic schema repair mechanism (retries with error feedback)
10. Resilience features (circuit breaker, retry policies)

The demo includes an extremely complex Person schema with:
- 7 nested models (PhoneNumber, Coordinates, Address, EmploymentDetails, etc.)
- 40+ fields with strict validation rules
- Regex patterns for names, emails, phone numbers, dates
- Range validations for age, salary, credit score
- Required list fields with min/max constraints
- Designed to trigger validation errors and demonstrate automatic repair

Setup:
    Create a .env file in the project root with:
        OPENAI_API_KEY=sk-...

    Or export environment variable:
        export OPENAI_API_KEY="sk-..."

Usage:
    python examples/comprehensive_demo.py
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Core imports
from parsec import (
    EnforcementEngine,
    JSONValidator,
    PydanticValidator,
    InMemoryCache,
    PromptTemplate,
    TemplateRegistry,
    TemplateManager,
    DatasetCollector,
)

# Adapter imports
from parsec.models.adapters import OpenAIAdapter

# Resilience imports
from parsec.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryPolicy,
    OperationType,
    ExponentialBackoff,
)

# Prompt analytics and testing
from parsec.prompts import (
    TemplateAnalytics,
    ABTest,
    Variant,
    TrafficSplitStrategy,
)


# ============================================================================
# 1. Define Pydantic Models for Complex Validation
# ============================================================================

class PhoneNumber(BaseModel):
    """Phone number with strict formatting."""
    country_code: str = Field(..., pattern=r"^\+\d{1,3}$", description="Country code with + prefix (e.g., +1)")
    area_code: str = Field(..., pattern=r"^\d{3}$", description="Exactly 3 digits")
    exchange: str = Field(..., pattern=r"^\d{3}$", description="Exactly 3 digits")
    number: str = Field(..., pattern=r"^\d{4}$", description="Exactly 4 digits")
    extension: Optional[str] = Field(None, pattern=r"^x\d{1,5}$", description="Extension starting with 'x'")


class Coordinates(BaseModel):
    """GPS coordinates with precise validation."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude between -90 and 90")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude between -180 and 180")
    altitude_meters: Optional[float] = Field(None, ge=-500.0, le=9000.0, description="Altitude in meters")
    accuracy_meters: float = Field(..., gt=0.0, le=100.0, description="GPS accuracy in meters, must be positive")


class Address(BaseModel):
    """Nested address model with complex validation."""
    street_number: int = Field(..., gt=0, le=99999, description="Street number (1-99999)")
    street_name: str = Field(..., min_length=2, max_length=100, description="Street name")
    street_type: str = Field(..., pattern=r"^(Street|Avenue|Boulevard|Road|Lane|Drive|Court|Place|Way)$", description="Must be one of: Street, Avenue, Boulevard, Road, Lane, Drive, Court, Place, Way")
    unit_number: Optional[str] = Field(None, pattern=r"^(Apt|Suite|Unit|#) [A-Z0-9]+$", description="Unit with type prefix")
    city: str = Field(..., min_length=2, max_length=50, description="City name")
    state: str = Field(..., pattern=r"^[A-Z]{2}$", description="Two-letter state code (uppercase)")
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$", description="5-digit ZIP or ZIP+4")
    country: str = Field(default="USA", pattern=r"^[A-Z]{3}$", description="3-letter country code (uppercase)")
    coordinates: Optional[Coordinates] = Field(None, description="GPS coordinates")
    is_primary: bool = Field(True, description="Whether this is the primary address")


class EmploymentDetails(BaseModel):
    """Employment information with date validation."""
    company_name: str = Field(..., min_length=2, max_length=100, description="Company name")
    job_title: str = Field(..., min_length=2, max_length=100, description="Job title")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date in YYYY-MM-DD format, or null if current")
    annual_salary_usd: float = Field(..., gt=0.0, le=10000000.0, description="Annual salary in USD")
    employment_type: str = Field(..., pattern=r"^(Full-time|Part-time|Contract|Freelance|Internship)$", description="Employment type")
    is_remote: bool = Field(False, description="Whether position is remote")
    department: str = Field(..., min_length=2, max_length=50, description="Department name")


class SocialMediaHandle(BaseModel):
    """Social media handle with platform-specific validation."""
    platform: str = Field(..., pattern=r"^(Twitter|LinkedIn|GitHub|Instagram|Facebook)$", description="Social media platform")
    username: str = Field(..., pattern=r"^@?[a-zA-Z0-9_-]{3,30}$", description="Username (3-30 chars, alphanumeric, _, -)")
    follower_count: Optional[int] = Field(None, ge=0, le=1000000000, description="Number of followers")
    verified: bool = Field(False, description="Whether account is verified")


class EducationRecord(BaseModel):
    """Education history with degree validation."""
    institution_name: str = Field(..., min_length=2, max_length=100, description="Institution name")
    degree_type: str = Field(..., pattern=r"^(High School|Associate|Bachelor|Master|Doctorate|Certificate)$", description="Degree type")
    major: str = Field(..., min_length=2, max_length=100, description="Major or field of study")
    graduation_date: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="Graduation date as YYYY-MM")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA on 4.0 scale")
    honors: Optional[str] = Field(None, pattern=r"^(Summa Cum Laude|Magna Cum Laude|Cum Laude)$", description="Academic honors")


class Person(BaseModel):
    """Highly complex person model designed to challenge validation."""
    # Basic identification
    first_name: str = Field(..., min_length=2, max_length=50, pattern=r"^[A-Z][a-z]+$", description="First name (capitalized, letters only)")
    middle_initial: Optional[str] = Field(None, pattern=r"^[A-Z]\.$", description="Middle initial with period (e.g., 'J.')")
    last_name: str = Field(..., min_length=2, max_length=50, pattern=r"^[A-Z][a-z]+(-[A-Z][a-z]+)?$", description="Last name (capitalized, may be hyphenated)")
    preferred_name: Optional[str] = Field(None, min_length=2, max_length=50, description="Preferred name or nickname")

    # Contact information
    email_primary: str = Field(..., pattern=r"^[a-z0-9._+-]+@[a-z0-9.-]+\.[a-z]{2,}$", description="Primary email (lowercase)")
    email_secondary: Optional[str] = Field(None, pattern=r"^[a-z0-9._+-]+@[a-z0-9.-]+\.[a-z]{2,}$", description="Secondary email")
    phone_primary: PhoneNumber = Field(..., description="Primary phone number")
    phone_secondary: Optional[PhoneNumber] = Field(None, description="Secondary phone number")

    # Demographics
    age: int = Field(..., ge=18, le=120, description="Age in years (must be 18+)")
    date_of_birth: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date of birth in YYYY-MM-DD format")
    gender: str = Field(..., pattern=r"^(Male|Female|Non-binary|Prefer not to say)$", description="Gender")
    nationality: str = Field(..., pattern=r"^[A-Z]{3}$", description="3-letter nationality code (uppercase)")

    # Addresses (at least one required)
    addresses: list[Address] = Field(..., min_length=1, max_length=5, description="List of addresses (1-5)")

    # Employment
    is_employed: bool = Field(..., description="Whether currently employed")
    current_employment: Optional[EmploymentDetails] = Field(None, description="Current employment details")
    previous_employments: list[EmploymentDetails] = Field(default_factory=list, max_length=10, description="Previous employment history")

    # Education (at least one required)
    education_history: list[EducationRecord] = Field(..., min_length=1, max_length=10, description="Education history (1-10 records)")

    # Social media
    social_media_handles: list[SocialMediaHandle] = Field(default_factory=list, max_length=10, description="Social media profiles")

    # Interests and metadata
    hobbies: list[str] = Field(default_factory=list, min_length=0, max_length=20, description="List of hobbies")
    languages_spoken: list[str] = Field(..., min_length=1, max_length=10, description="Languages spoken (ISO codes)")
    security_clearance_level: Optional[str] = Field(None, pattern=r"^(Confidential|Secret|Top Secret)$", description="Security clearance")

    # Financial
    annual_income_usd: float = Field(..., gt=0.0, le=100000000.0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")

    # Metadata
    emergency_contact_name: str = Field(..., min_length=2, max_length=100, description="Emergency contact full name")
    emergency_contact_phone: PhoneNumber = Field(..., description="Emergency contact phone")
    emergency_contact_relationship: str = Field(..., pattern=r"^(Spouse|Parent|Sibling|Child|Friend|Other)$", description="Relationship to emergency contact")


class Product(BaseModel):
    """Product information model."""
    name: str
    price: float = Field(..., gt=0)
    category: str
    in_stock: bool = Field(True)
    tags: list[str] = Field(default_factory=list)


# ============================================================================
# 2. Setup OpenAI Adapter
# ============================================================================

def setup_adapter():
    """Create OpenAI adapter."""
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found. Set it in .env file or as environment variable.")

    print("✓ OpenAI adapter configured")
    return OpenAIAdapter(api_key=openai_key, model="gpt-4o-mini")


# ============================================================================
# 3. Setup Resilience Features
# ============================================================================

def setup_circuit_breaker():
    """Configure circuit breaker for fault tolerance."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=60.0
    )
    return CircuitBreaker(name="llm_breaker", config=config)


def setup_retry_policy():
    """Configure retry policy with exponential backoff."""
    backoff = ExponentialBackoff(
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True
    )

    return RetryPolicy(
        operation_type=OperationType.LLM_GENERATION,
        max_retries=3,
        backoff_strategy=backoff,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )


# ============================================================================
# 4. Setup Prompt Templates with Versioning
# ============================================================================

def setup_templates():
    """Create versioned prompt templates."""
    registry = TemplateRegistry()

    # Template v1.0.0 - Basic person extraction
    template_v1 = PromptTemplate(
        name="extract_person",
        template="Extract person information from the following text:\n\n{text}\n\nReturn as JSON.",
        variables={"text": str},
        required=["text"]
    )
    registry.register(template_v1, "1.0.0")

    # Template v2.0.0 - Enhanced with validation rules
    template_v2 = PromptTemplate(
        name="extract_person",
        template="""Extract person information from the following text:

{text}

Validation Rules:
- Email must be valid format
- Age must be 0-150
- State must be 2-letter code
- ZIP must be 5 digits

Return as JSON with fields: name, age, email, address (street, city, state, zip_code), hobbies, is_employed.""",
        variables={"text": str},
        required=["text"]
    )
    registry.register(template_v2, "2.0.0")

    # Product extraction template
    product_template = PromptTemplate(
        name="extract_product",
        template="Extract product information from: {description}\n\nReturn as JSON with: name, price, category, in_stock, tags.",
        variables={"description": str},
        required=["description"]
    )
    registry.register(product_template, "1.0.0")

    # A/B testing variants
    variant_a = PromptTemplate(
        name="extract_person_variant_a",
        template="Extract all details about the person from: {text}\n\nBe thorough and accurate.",
        variables={"text": str},
        required=["text"]
    )
    registry.register(variant_a, "1.0.0")

    variant_b = PromptTemplate(
        name="extract_person_variant_b",
        template="Parse the following text and extract person data: {text}\n\nFocus on accuracy.",
        variables={"text": str},
        required=["text"]
    )
    registry.register(variant_b, "1.0.0")

    return registry


# ============================================================================
# 5. Demonstration Functions
# ============================================================================

async def demo_basic_enforcement(engine, validator):
    """Demo 1: Basic enforcement with JSON schema."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Enforcement with JSON Schema")
    print("="*80)

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age"]
    }

    prompt = "Extract information: Sarah Johnson is 28 years old and lives in Seattle."

    result = await engine.enforce(prompt, schema)

    print(f"Success: {result.success}")
    print(f"Data: {json.dumps(result.data, indent=2)}")
    print(f"Retries: {result.retry_count}")
    print(f"Validation status: {result.validation.status}")
    if result.validation.errors:
        print(f"Validation errors: {len(result.validation.errors)}")
        for error in result.validation.errors:
            print(f"  - {error.path}: {error.message}")
    else:
        print("Validation errors: None")


async def demo_pydantic_validation(engine):
    """Demo 2: Pydantic model validation with complex nesting - designed to trigger errors."""
    print("\n" + "="*80)
    print("DEMO 2: Complex Pydantic Validation with Automatic Repair")
    print("="*80)
    print("\nThis demo uses an extremely complex schema with:")
    print("  - Deeply nested models (PhoneNumber, Address, Coordinates, etc.)")
    print("  - Strict regex patterns for names, emails, dates, phone numbers")
    print("  - Range validations for age, salary, credit score")
    print("  - Required list fields with min/max constraints")
    print("  - Multiple interdependent fields")
    print("\nThe AI will likely make mistakes, triggering the repair mechanism!\n")

    prompt = """Extract comprehensive person data from this narrative:

    Sarah Johnson-Williams (goes by Sally) was born on March 15th, 1985, making her 39 years old.
    She's a Female American citizen. Her primary email is sarah.j.williams@techcorp.com and you can
    also reach her at sally.personal@gmail.com. Her main phone is 1-415-555-1234 extension 5678 and
    her mobile is +1 (415) 555-9876.

    She lives at 456 Market Street, Apartment 12B in San Francisco, California, 94102-1234, USA.
    The coordinates are roughly 37.7749 latitude and -122.4194 longitude with 50 meter accuracy.
    This is her primary residence.

    Sarah currently works at TechCorp Inc as a Senior Software Engineer in the Engineering Department.
    She started on 2020-01-15 and earns $185,000 per year as a full-time remote employee.

    Previously, she worked at StartupXYZ as a Software Developer (Part-time) from 2018-06-01 to
    2019-12-31, making $95,000 annually in the Product department.

    She has a Bachelor degree in Computer Science from Stanford University, graduated May 2007
    with a 3.8 GPA and Magna Cum Laude honors. She also has a Master degree in Artificial
    Intelligence from MIT, graduating in 2009-06 with a perfect 4.0 GPA.

    On social media, you can find her on LinkedIn as @sarah-j-williams with 5000 followers (verified),
    Twitter as @sarahcodes with 12000 followers, and GitHub as @sjohnsonwilliams with 3500 followers.

    She speaks English, Spanish, and French. Her hobbies include rock climbing, photography, machine
    learning, open source contribution, and playing piano. She holds a Secret security clearance.

    Her annual income is $185,000 and her credit score is 780. In case of emergency, contact her
    spouse Michael Williams at +1-415-555-4321.
    """

    result = await engine.enforce(prompt, Person)

    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS:")
    print(f"{'='*80}")
    print(f"Success: {result.success}")
    print(f"Retry count: {result.retry_count}")
    print(f"Validation status: {result.validation.status}")

    if result.validation.errors:
        print(f"\n⚠ Validation errors encountered: {len(result.validation.errors)}")
        print("Showing first 10 errors:")
        for i, error in enumerate(result.validation.errors[:10]):
            print(f"  {i+1}. {error.path}: {error.message}")
        if len(result.validation.errors) > 10:
            print(f"  ... and {len(result.validation.errors) - 10} more errors")
        print(f"\n✓ Repair mechanism was triggered! Retried {result.retry_count} time(s)")
    else:
        print("✓ No validation errors - AI got it right on first try!")

    print(f"\n{'='*80}")
    print(f"EXTRACTED DATA (showing structure):")
    print(f"{'='*80}")

    # Show the data in a more readable format
    data = result.data
    print(f"Name: {data.get('first_name', 'N/A')} {data.get('middle_initial', '')} {data.get('last_name', 'N/A')}")
    print(f"Age: {data.get('age', 'N/A')}")
    print(f"Email: {data.get('email_primary', 'N/A')}")
    print(f"Phone: {data.get('phone_primary', {})}")
    print(f"Addresses: {len(data.get('addresses', []))} address(es)")
    print(f"Employment: {'Yes' if data.get('is_employed') else 'No'}")
    print(f"Education: {len(data.get('education_history', []))} record(s)")
    print(f"Social Media: {len(data.get('social_media_handles', []))} account(s)")

    print(f"\nFull JSON output:")
    print(json.dumps(result.data, indent=2))


async def demo_caching(engine, cache):
    """Demo 3: Caching to reduce redundant API calls."""
    print("\n" + "="*80)
    print("DEMO 3: Caching with Statistics")
    print("="*80)

    schema = {"type": "object", "properties": {"product": {"type": "string"}}}
    prompt = "Extract product: iPhone 15 Pro"

    # First call - cache miss
    print("\nFirst call (cache miss)...")
    result1 = await engine.enforce(prompt, schema)
    stats1 = cache.get_stats()
    print(f"Cache stats: {stats1}")

    # Second identical call - cache hit
    print("\nSecond identical call (cache hit)...")
    result2 = await engine.enforce(prompt, schema)
    stats2 = cache.get_stats()
    print(f"Cache stats: {stats2}")

    # Different prompt - cache miss
    print("\nDifferent prompt (cache miss)...")
    result3 = await engine.enforce("Extract product: Samsung Galaxy", schema)
    stats3 = cache.get_stats()
    print(f"Cache stats: {stats3}")

    print(f"\nFinal cache performance:")
    print(f"  Hits: {stats3['hits']}")
    print(f"  Misses: {stats3['misses']}")
    print(f"  Hit Rate: {stats3['hit_rate']}")


async def demo_template_manager(manager, registry):
    """Demo 4: Template manager with versioning."""
    print("\n" + "="*80)
    print("DEMO 4: Template Manager with Versioning")
    print("="*80)

    # Use version 1.0.0
    print("\nUsing template v1.0.0:")
    result_v1 = await manager.enforce_with_template(
        template_name="extract_person",
        version="1.0.0",
        variables={"text": "Alice Brown, age 42, alice@test.com"},
        schema=Person
    )
    if result_v1:
        print(f"Result: {json.dumps(result_v1.data, indent=2)}")
    else:
        print("⚠ Template v1.0.0 failed to produce result")

    # Use latest version (2.0.0)
    print("\nUsing latest template (v2.0.0):")
    result_v2 = await manager.enforce_with_template(
        template_name="extract_person",
        variables={"text": "Bob Wilson, age 55, bob@example.org, lives in NYC"},
        schema=Person
    )
    if result_v2:
        print(f"Result: {json.dumps(result_v2.data, indent=2)}")
    else:
        print("⚠ Template v2.0.0 failed to produce result")


async def demo_template_persistence(registry):
    """Demo 5: Save and load templates from disk."""
    print("\n" + "="*80)
    print("DEMO 5: Template Persistence (YAML)")
    print("="*80)

    # Save templates
    output_path = Path("./templates_demo.yaml")
    registry.save_to_disk(str(output_path))
    print(f"✓ Saved templates to {output_path}")

    # Load into new registry
    new_registry = TemplateRegistry()
    new_registry.load_from_disk(str(output_path))

    template_names = new_registry.list_templates()
    print(f"\n✓ Loaded {len(template_names)} template(s):")
    for name in template_names:
        versions = new_registry.list_versions(name)
        print(f"  - {name}: versions {', '.join(versions)}")

    # Cleanup
    output_path.unlink()
    print(f"\n✓ Cleaned up {output_path}")


async def demo_dataset_collection(collector):
    """Demo 6: Dataset collection for training."""
    print("\n" + "="*80)
    print("DEMO 6: Dataset Collection for Training")
    print("="*80)

    # Create engine with collector
    adapter = setup_adapter()
    validator = PydanticValidator()
    engine = EnforcementEngine(adapter, validator, collector=collector)

    # Run multiple enforcements (data gets collected automatically)
    samples = [
        "Emma Davis, 29, emma.davis@email.com",
        "Michael Chen, 41, mike@company.com",
        "Lisa Anderson, 33, lisa.a@domain.net"
    ]

    print(f"\nCollecting {len(samples)} samples...")
    for text in samples:
        await engine.enforce(f"Extract: {text}", Person)

    # Write collected data to disk
    collector.close()

    print(f"\n✓ Collected and saved {collector.examples_written} samples")
    print(f"✓ Saved to: {collector.output_path}")

    # Show buffer info (note: buffer is cleared after close())
    print(f"\nData has been written to disk in {collector.format.upper()} format")


async def demo_template_analytics(analytics):
    """Demo 7: Template analytics and metrics."""
    print("\n" + "="*80)
    print("DEMO 7: Template Analytics")
    print("="*80)

    # Record some metrics
    analytics.record_result(
        template_name="extract_person",
        version="2.0.0",
        success=True,
        tokens_used=45,
        latency_ms=150.5,
        retry_count=0
    )

    analytics.record_result(
        template_name="extract_person",
        version="2.0.0",
        success=True,
        tokens_used=42,
        latency_ms=120.3,
        retry_count=0
    )

    analytics.record_result(
        template_name="extract_product",
        version="1.0.0",
        success=True,
        tokens_used=30,
        latency_ms=95.0,
        retry_count=1
    )

    # Get metrics
    metrics = analytics.get_metrics("extract_person", "2.0.0")
    print(f"\nMetrics for extract_person v2.0.0:")
    print(f"  Total calls: {metrics.total_calls}")
    print(f"  Success rate: {metrics.success_rate:.2%}")
    print(f"  Avg latency: {metrics.average_latency_ms:.2f}ms")
    print(f"  Avg tokens: {metrics.average_tokens:.1f}")
    print(f"  Avg retries: {metrics.average_retries:.2f}")


async def demo_ab_testing():
    """Demo 8: A/B testing for prompt optimization."""
    print("\n" + "="*80)
    print("DEMO 8: A/B Testing")
    print("="*80)

    # Create analytics instance for the test
    test_analytics = TemplateAnalytics()

    # Create variants
    variant_a = Variant(
        template_name="extract_person_variant_a",
        version="1.0.0",
        weight=0.5
    )

    variant_b = Variant(
        template_name="extract_person_variant_b",
        version="1.0.0",
        weight=0.5
    )

    # Create A/B test
    ab_test = ABTest(
        test_name="person_extraction_test",
        variants=[variant_a, variant_b],
        analytics=test_analytics,
        strategy=TrafficSplitStrategy.WEIGHTED
    )

    # Simulate traffic and record results
    print("\nSimulating 10 requests with 50/50 split:")
    for i in range(10):
        variant = ab_test.select_variant()
        print(f"  Request {i+1}: {variant.template_name}:{variant.version}")

        # Simulate execution and record in analytics
        if variant == variant_a:
            test_analytics.record_result(
                template_name=variant.template_name,
                version=variant.version,
                success=True,
                tokens_used=40,
                latency_ms=100 + i,
                retry_count=0
            )
        else:
            test_analytics.record_result(
                template_name=variant.template_name,
                version=variant.version,
                success=True,
                tokens_used=38,
                latency_ms=90 + i,
                retry_count=0
            )

    # Get results
    result = ab_test.get_results()
    print(f"\nA/B Test Results:")
    print(f"  Total samples: {result.sample_size}")
    print(f"  Is significant: {result.is_significant}")
    if result.winner:
        print(f"  Winner: {result.winner.template_name}:{result.winner.version}")
        print(f"  Confidence: {result.confidence:.2%}")
    else:
        print(f"  No clear winner yet (need more samples or difference not significant)")

    print(f"\nMetrics by variant:")
    for variant_key, metrics in result.metrics_by_variant.items():
        print(f"  {variant_key}:")
        print(f"    Total calls: {metrics.total_calls}")
        print(f"    Success rate: {metrics.success_rate:.2%}")
        print(f"    Avg latency: {metrics.average_latency_ms:.2f}ms")


async def demo_schema_repair(engine):
    """Demo 9: Schema repair mechanism with intentionally difficult validation."""
    print("\n" + "="*80)
    print("DEMO 9: Schema Repair Mechanism")
    print("="*80)
    print("\nDemonstrating automatic repair when LLM produces invalid output.")
    print("Using intentionally ambiguous and poorly formatted data...\n")

    # EXTREMELY challenging prompt with many format violations and ambiguities
    prompt = """Extract person data from this messy text:

    name is john doe jr but everyone calls him johnny. he's like 29 or 30 yrs old?
    born sometime in november 1994, i think the 23rd. he's a guy. american.

    emails are JOHN.DOE@WORK.COM and jdoe_personal@yahoo

    phone numbers: work is 555.1234 (no area code?), cell: 4155559999

    lives on pine st, number... 12 or 21? in apartment B (or was it C?),
    somewhere in portland OR, zip is 97201 i think

    works at some startup, been there since '22. makes around 80k.
    job title: "engineer" or "developer"? part time i think. or was it full time?

    went to college somewhere, got a degree in 2016 or 2017. gpa was decent.

    twitter: @johndoe123456789012345678901234567890 (that handle is too long!)
    linkedin: johndoe

    speaks english. maybe spanish? hobbies unclear.

    credit score 500 (that's below minimum!)
    salary: -50000 (negative salary!)

    emergency contact: jane (relationship unknown) phone: 5551234 (incomplete!)
    """

    print("This prompt contains many validation violations:")
    print("  - Ambiguous/missing data")
    print("  - Wrong formats (emails without domains, incomplete phones)")
    print("  - Values outside valid ranges (credit score 500, negative salary)")
    print("  - Too long strings (Twitter handle)")
    print("  - Missing required fields")
    print("\nRunning enforcement with max_retries=5 to allow repair attempts...\n")

    # Create a fresh engine with higher retry count for this demo
    adapter = setup_adapter()
    validator = PydanticValidator()
    repair_engine = EnforcementEngine(
        adapter=adapter,
        validator=validator,
        max_retries=5  # More retries to demonstrate repair
    )

    result = await repair_engine.enforce(prompt, Person)

    print(f"\n{'='*80}")
    print(f"REPAIR MECHANISM RESULTS:")
    print(f"{'='*80}")
    print(f"Final success: {result.success}")
    print(f"Total retry attempts: {result.retry_count}")
    print(f"Validation status: {result.validation.status}")

    if result.retry_count > 0:
        print(f"\n✓ REPAIR MECHANISM ACTIVATED!")
        print(f"  The LLM made mistakes on the first attempt(s)")
        print(f"  System automatically retried with error feedback")
        print(f"  After {result.retry_count} repair attempt(s), {'succeeded' if result.success else 'still has errors'}")

    if result.validation.errors:
        print(f"\n⚠ Remaining validation errors: {len(result.validation.errors)}")
        for i, error in enumerate(result.validation.errors[:5]):
            print(f"  {i+1}. {error.path}: {error.message}")
    else:
        print("\n✓ All validation errors resolved!")

    print(f"\n{'='*80}")
    print(f"Key fields extracted:")
    print(f"{'='*80}")
    data = result.data
    print(f"Name: {data.get('first_name', '?')} {data.get('last_name', '?')}")
    print(f"Preferred: {data.get('preferred_name', 'N/A')}")
    print(f"Primary Email: {data.get('email_primary', '?')}")
    if data.get('phone_primary'):
        phone = data['phone_primary']
        print(f"Phone: {phone.get('country_code', '?')}-{phone.get('area_code', '?')}-{phone.get('exchange', '?')}-{phone.get('number', '?')}")
    print(f"Currently Employed: {data.get('is_employed', '?')}")


async def demo_circuit_breaker():
    """Demo 10: Circuit breaker pattern."""
    print("\n" + "="*80)
    print("DEMO 10: Circuit Breaker")
    print("="*80)

    breaker = setup_circuit_breaker()

    print(f"\nInitial state: {breaker.state.name}")
    print(f"Failure threshold: {breaker.config.failure_threshold}")
    print(f"Success threshold: {breaker.config.success_threshold}")
    print(f"Timeout: {breaker.config.timeout}s")

    # Demonstrate circuit breaker state
    state = breaker.get_state()
    print(f"\nCircuit breaker state:")
    print(f"  Name: {state['name']}")
    print(f"  State: {state['state']}")
    print(f"  Failure count: {state['failure_count']}")
    print(f"  Success count: {state['success_count']}")

    print(f"\nCircuit breaker can be used to wrap async calls:")
    print(f"  - Tracks failures automatically")
    print(f"  - Opens circuit after {breaker.config.failure_threshold} failures")
    print(f"  - Blocks calls when open to fail fast")
    print(f"  - Enters half-open state after {breaker.config.timeout}s timeout")
    print(f"  - Closes after {breaker.config.success_threshold} successes in half-open")


# ============================================================================
# 6. Main Execution
# ============================================================================

async def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print(" PARSEC COMPREHENSIVE FEATURE DEMONSTRATION")
    print("="*80)

    # Setup components
    print("\n[Setup Phase]")
    adapter = setup_adapter()
    print(f"Using adapter: {adapter.__class__.__name__}")

    # Setup validators
    json_validator = JSONValidator()
    pydantic_validator = PydanticValidator()

    # Setup cache
    cache = InMemoryCache(max_size=100, default_ttl=3600)

    # Setup dataset collector
    output_dir = Path("./training_data_demo")
    output_dir.mkdir(exist_ok=True)
    collector = DatasetCollector(
        output_path=str(output_dir / "dataset.jsonl"),
        format="jsonl",
    )

    # Setup templates and registry
    registry = setup_templates()

    # Setup analytics
    analytics = TemplateAnalytics()

    # Create engines
    json_engine = EnforcementEngine(
        adapter=adapter,
        validator=json_validator,
        max_retries=3,
        cache=cache
    )

    pydantic_engine = EnforcementEngine(
        adapter=adapter,
        validator=pydantic_validator,
        max_retries=3,
        cache=cache
    )

    # Setup template manager
    manager = TemplateManager(registry, pydantic_engine)

    print("\n[Running Demonstrations]")

    # Run all demos
    await demo_basic_enforcement(json_engine, json_validator)
    await demo_pydantic_validation(pydantic_engine)
    await demo_caching(json_engine, cache)
    await demo_template_manager(manager, registry)
    await demo_template_persistence(registry)
    await demo_dataset_collection(collector)
    await demo_template_analytics(analytics)
    await demo_ab_testing()
    await demo_schema_repair(pydantic_engine)
    await demo_circuit_breaker()

    # Cleanup
    print("\n" + "="*80)
    print("[Cleanup]")
    print("="*80)
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"✓ Cleaned up {output_dir}")

    print("\n" + "="*80)
    print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nFeatures demonstrated:")
    features = [
        "1. Basic enforcement with JSON schema",
        "2. Complex Pydantic validation with nested models",
        "3. Caching with hit/miss statistics",
        "4. Template manager with versioning",
        "5. Template persistence (YAML save/load)",
        "6. Dataset collection for training",
        "7. Template analytics and metrics",
        "8. A/B testing for prompt optimization",
        "9. Automatic schema repair mechanism",
        "10. Circuit breaker pattern"
    ]
    for feature in features:
        print(f"  ✓ {feature}")


if __name__ == "__main__":
    asyncio.run(main())
