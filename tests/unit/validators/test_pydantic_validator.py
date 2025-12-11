"""Tests for Pydantic validator."""
import pytest
from pydantic import BaseModel, Field
from typing import Optional, List

from parsec.validators.pydantic_validator import PydanticValidator
from parsec.validators.base_validator import ValidationStatus


class SimplePerson(BaseModel):
    """Simple person model for testing."""
    name: str
    age: int


class PersonWithOptional(BaseModel):
    """Person model with optional fields."""
    name: str
    age: int
    email: Optional[str] = None


class PersonWithValidation(BaseModel):
    """Person model with field validation."""
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')


class NestedAddress(BaseModel):
    """Nested address model."""
    street: str
    city: str
    zipcode: str


class PersonWithAddress(BaseModel):
    """Person with nested address."""
    name: str
    age: int
    address: NestedAddress


class PersonWithList(BaseModel):
    """Person with list of hobbies."""
    name: str
    hobbies: List[str]


class TestPydanticValidator:
    """Test PydanticValidator functionality."""

    def test_create_validator(self):
        """Test creating a Pydantic validator."""
        validator = PydanticValidator()
        assert validator is not None

    def test_validate_simple_valid_output(self):
        """Test validating simple valid output."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30}'

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.VALID
        assert result.parsed_output == {"name": "John Doe", "age": 30}
        assert result.raw_output == output
        assert len(result.errors) == 0

    def test_validate_with_optional_field_present(self):
        """Test validating with optional field present."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'

        result = validator.validate(output, PersonWithOptional)

        assert result.status == ValidationStatus.VALID
        assert result.parsed_output["email"] == "john@example.com"

    def test_validate_with_optional_field_missing(self):
        """Test validating with optional field missing."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30}'

        result = validator.validate(output, PersonWithOptional)

        assert result.status == ValidationStatus.VALID
        assert result.parsed_output["email"] is None

    def test_validate_missing_required_field(self):
        """Test validating with missing required field."""
        validator = PydanticValidator()
        output = '{"name": "John Doe"}'  # Missing age

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.INVALID
        assert len(result.errors) > 0
        assert any("age" in error.path for error in result.errors)

    def test_validate_wrong_type(self):
        """Test validating with wrong field type."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": "thirty"}'  # Age should be int

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.INVALID
        assert len(result.errors) > 0
        assert any("age" in error.path for error in result.errors)

    def test_validate_invalid_json(self):
        """Test validating with invalid JSON."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30'  # Missing closing brace

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.INVALID
        assert len(result.errors) > 0
        assert result.errors[0].message == "Invalid JSON format"

    def test_validate_field_validation_passes(self):
        """Test field-level validation that passes."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'

        result = validator.validate(output, PersonWithValidation)

        assert result.status == ValidationStatus.VALID

    def test_validate_field_validation_fails_min_length(self):
        """Test field validation fails on min length."""
        validator = PydanticValidator()
        output = '{"name": "", "age": 30, "email": "john@example.com"}'

        result = validator.validate(output, PersonWithValidation)

        assert result.status == ValidationStatus.INVALID
        assert any("name" in error.path for error in result.errors)

    def test_validate_field_validation_fails_range(self):
        """Test field validation fails on range."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 200, "email": "john@example.com"}'

        result = validator.validate(output, PersonWithValidation)

        assert result.status == ValidationStatus.INVALID
        assert any("age" in error.path for error in result.errors)

    def test_validate_field_validation_fails_pattern(self):
        """Test field validation fails on pattern."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30, "email": "invalid-email"}'

        result = validator.validate(output, PersonWithValidation)

        assert result.status == ValidationStatus.INVALID
        assert any("email" in error.path for error in result.errors)

    def test_validate_nested_model_valid(self):
        """Test validating nested model that's valid."""
        validator = PydanticValidator()
        output = '''
        {
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "zipcode": "12345"
            }
        }
        '''

        result = validator.validate(output, PersonWithAddress)

        assert result.status == ValidationStatus.VALID
        assert result.parsed_output["address"]["street"] == "123 Main St"

    def test_validate_nested_model_missing_field(self):
        """Test validating nested model with missing field."""
        validator = PydanticValidator()
        output = '''
        {
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "Springfield"
            }
        }
        '''  # Missing zipcode

        result = validator.validate(output, PersonWithAddress)

        assert result.status == ValidationStatus.INVALID
        assert any("zipcode" in error.path for error in result.errors)

    def test_validate_list_field_valid(self):
        """Test validating list field that's valid."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "hobbies": ["reading", "cycling", "cooking"]}'

        result = validator.validate(output, PersonWithList)

        assert result.status == ValidationStatus.VALID
        assert len(result.parsed_output["hobbies"]) == 3

    def test_validate_list_field_empty(self):
        """Test validating empty list field."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "hobbies": []}'

        result = validator.validate(output, PersonWithList)

        assert result.status == ValidationStatus.VALID
        assert result.parsed_output["hobbies"] == []

    def test_validate_list_field_wrong_type(self):
        """Test validating list field with wrong type."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "hobbies": "reading"}'  # Should be list

        result = validator.validate(output, PersonWithList)

        assert result.status == ValidationStatus.INVALID
        assert any("hobbies" in error.path for error in result.errors)

    def test_validate_extra_fields_ignored(self):
        """Test that extra fields are ignored by default."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30, "extra": "ignored"}'

        result = validator.validate(output, SimplePerson)

        # Should still be valid, extra field ignored
        assert result.status == ValidationStatus.VALID
        assert "extra" not in result.parsed_output

    def test_repair_invalid_json(self):
        """Test repair functionality."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30'  # Missing closing brace

        repaired = validator.repair(output, [])

        # Repair attempt may or may not succeed - just verify it returns a string
        assert isinstance(repaired, str)

    def test_validate_multiple_errors(self):
        """Test validating with multiple errors."""
        validator = PydanticValidator()
        output = '{}'  # Missing both name and age

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.INVALID
        assert len(result.errors) >= 2  # At least name and age missing

    def test_validate_and_repair(self):
        """Test validate_and_repair method (inherited from BaseValidator)."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30}'

        result = validator.validate_and_repair(output, SimplePerson)

        assert result.status == ValidationStatus.VALID
        assert result.parsed_output == {"name": "John Doe", "age": 30}

    def test_error_contains_expected_fields(self):
        """Test that validation errors contain expected fields."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": "thirty"}'

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.INVALID
        error = result.errors[0]
        assert error.path is not None
        assert error.message is not None
        assert error.expected is not None
        assert error.severity == "error"

    def test_validate_preserves_raw_output(self):
        """Test that raw output is preserved in result."""
        validator = PydanticValidator()
        output = '{"name": "John Doe", "age": 30}'

        result = validator.validate(output, SimplePerson)

        assert result.raw_output == output

    def test_validate_null_value(self):
        """Test validating with null value for required field."""
        validator = PydanticValidator()
        output = '{"name": null, "age": 30}'

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.INVALID
        assert any("name" in error.path for error in result.errors)

    def test_validate_whitespace_in_json(self):
        """Test validating JSON with whitespace."""
        validator = PydanticValidator()
        output = '''
        {
            "name": "John Doe",
            "age": 30
        }
        '''

        result = validator.validate(output, SimplePerson)

        assert result.status == ValidationStatus.VALID
