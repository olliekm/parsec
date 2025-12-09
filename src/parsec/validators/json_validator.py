from typing import Any, Dict, List, Optional
from .base_validator import BaseValidator, ValidationResult, ValidationStatus, ValidationError
from .repair_utils import JSONRepairUtils
import jsonschema
import json


class JSONValidator(BaseValidator):
    """Validator that checks if the output is valid JSON and conforms to a given schema."""

    def __init__(self):
        self.validator = jsonschema.Draft7Validator

    def validate(self, output: str, schema: Dict[str, Any]) -> ValidationResult:
        errors = []

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                path="",
                message="Invalid JSON format",
                expected="Valid JSON",
                actual=str(e),
                severity="error"
            ))
            return ValidationResult(
                status=ValidationStatus.INVALID,
                errors=errors,
                raw_output=output
            )
        
        schema_validator = self.validator(schema)
        schema_errors = list(schema_validator.iter_errors(parsed))

        if not schema_errors:
            return ValidationResult(
                status=ValidationStatus.VALID,
                parsed_output=parsed,
                raw_output=output
            )
        
        for err in schema_errors:
            # Build proper JSON path: $.field.nested instead of $.$field$.$nested
            if err.path:
                path = "$." + ".".join(str(p) for p in err.path)
            else:
                path = "$"
            errors.append(ValidationError(
                path=path,
                message=err.message,
                expected=err.schema.get("type", "unknown"),
                actual=type(err.instance).__name__,
                severity="error"
            ))

        return ValidationResult(
            status=ValidationStatus.INVALID,
            errors=errors,
            raw_output=output,
            parsed_output=parsed
        )
        
    def repair(self, output: str, errors: List[ValidationError]) -> str:
        """Repair common JSON issues using shared repair utilities."""
        return JSONRepairUtils.repair(output)