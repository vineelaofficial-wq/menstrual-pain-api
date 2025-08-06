"""
Menstrual Pain Predictor Module

A comprehensive system for predicting the most likely underlying cause of period pain
based on patient questionnaire responses.

Author: Medical Software Engineer
"""

from typing import Dict, List, Optional, Union, Any
import json
from dataclasses import dataclass
from enum import Enum


class PainCause(Enum):
    """Enumeration of possible pain causes"""
    PRIMARY_DYSMENORRHEA = "primary_dysmenorrhea"
    ENDOMETRIOSIS = "endometriosis"
    FIBROIDS = "fibroids"
    PCOS_RELATED = "pcos_related"
    PID = "pid"
    ADENOMYOSIS = "adenomyosis"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class PatientData:
    """Data structure for patient questionnaire responses"""
    # Demographics
    age: int
    sex_assigned_at_birth: str  # "female", "male", "intersex"
    
    # Physical measurements
    height_cm: float
    weight_kg: float
    
    # Menstrual history
    age_of_menarche: int
    cycle_regularity: str  # "regular", "irregular", "very_irregular"
    period_duration_days: int
    
    # Pain characteristics
    pain_frequency: str  # "every_cycle", "most_cycles", "some_cycles", "rarely"
    pain_severity: int  # 1-10 scale
    pain_location: List[str]  # ["lower_abdomen", "back", "thighs", "pelvis"]
    pain_duration_days: int
    
    # Bleeding characteristics
    heavy_bleeding: bool
    clotting: bool
    
    # Associated symptoms
    pain_during_sex: bool
    bowel_pain: bool
    nausea: bool
    vomiting: bool
    diarrhea: bool
    fatigue: bool
    headache: bool
    bloating: bool
    
    # Medical history
    prior_diagnoses: List[str]  # ["pcos", "fibroids", "pid", "endometriosis", etc.]
    medical_history: List[str]  # ["diabetes", "autoimmune", "fertility_issues", etc.]
    family_history_reproductive: List[str]  # ["endometriosis", "fibroids", "pcos", etc.]

    # Optional fields (must come last)
    unusual_discharge: bool = False # Added
    fever: bool = False # Added
    gender_identity: Optional[str] = None
    race_ethnicity: Optional[str] = None


@dataclass
class PredictionResult:
    """Result structure for pain cause prediction"""
    predicted_cause: PainCause
    confidence_level: ConfidenceLevel
    clinical_notes: str
    red_flags: List[str]
    scores: Dict[str, float]  # Scores for each condition


# Symptom-to-condition mapping reference
SYMPTOM_CONDITION_MAPPING = {
    "primary_dysmenorrhea": {
        "key_symptoms": [
            "cramping_pain_lower_abdomen",
            "pain_starts_with_period",
            "pain_subsides_2_3_days",
            "nausea",
            "fatigue",
            "headache"
        ],
        "pain_characteristics": {
            "location": ["lower_abdomen", "back", "thighs"],
            "type": "cramping",
            "timing": "with_period",
            "severity": "mild_to_moderate"
        },
        "age_range": "teens_to_twenties",
        "associated_factors": ["no_underlying_pathology"]
    },
    
    "endometriosis": {
        "key_symptoms": [
            "chronic_pelvic_pain",
            "pain_during_sex",
            "pain_before_and_during_period",
            "heavy_bleeding",
            "infertility"
        ],
        "pain_characteristics": {
            "location": ["pelvis", "lower_abdomen"],
            "type": "sharp_stabbing",
            "timing": "chronic_cyclical",
            "severity": "moderate_to_severe"
        },
        "age_range": "reproductive_years",
        "associated_factors": ["family_history", "delayed_childbearing"]
    },
    
    "fibroids": {
        "key_symptoms": [
            "heavy_menstrual_bleeding",
            "prolonged_periods",
            "pelvic_pressure",
            "frequent_urination",
            "back_pain"
        ],
        "pain_characteristics": {
            "location": ["lower_abdomen", "back", "pelvis"],
            "type": "pressure_cramping",
            "timing": "during_period",
            "severity": "moderate"
        },
        "age_range": "thirties_forties",
        "associated_factors": ["african_american_ethnicity", "family_history"]
    },
    
    "pcos_related": {
        "key_symptoms": [
            "irregular_periods",
            "pelvic_pain",
            "bloating",
            "weight_gain",
            "acne"
        ],
        "pain_characteristics": {
            "location": ["pelvis", "lower_abdomen"],
            "type": "dull_aching",
            "timing": "irregular",
            "severity": "mild_to_moderate"
        },
        "age_range": "reproductive_years",
        "associated_factors": ["insulin_resistance", "obesity", "family_history"]
    },
    
    "pid": {
        "key_symptoms": [
            "pelvic_pain",
            "unusual_discharge",
            "fever",
            "pain_during_sex",
            "irregular_bleeding"
        ],
        "pain_characteristics": {
            "location": ["pelvis", "lower_abdomen"],
            "type": "constant_aching",
            "timing": "continuous",
            "severity": "moderate_to_severe"
        },
        "age_range": "sexually_active",
        "associated_factors": ["multiple_partners", "std_history", "recent_procedure"]
    },
    
    "adenomyosis": {
        "key_symptoms": [
            "severe_menstrual_cramps",
            "heavy_prolonged_bleeding",
            "enlarged_uterus",
            "chronic_pelvic_pain"
        ],
        "pain_characteristics": {
            "location": ["pelvis", "lower_abdomen"],
            "type": "sharp_knifelike",
            "timing": "during_period",
            "severity": "severe"
        },
        "age_range": "thirties_forties",
        "associated_factors": ["previous_pregnancies", "uterine_surgery"]
    }
}


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate BMI from height and weight"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def normalize_pain_severity(severity: int) -> float:
    """Normalize pain severity to 0-1 scale"""
    return min(max(severity, 1), 10) / 10.0


def preprocess_patient_data(data: PatientData) -> Dict[str, Any]:
    """Preprocess patient data for analysis"""
    processed = {
        "bmi": calculate_bmi(data.height_cm, data.weight_kg),
        "normalized_pain_severity": normalize_pain_severity(data.pain_severity),
        "age_category": categorize_age(data.age),
        "cycle_regularity_score": score_cycle_regularity(data.cycle_regularity),
        "pain_frequency_score": score_pain_frequency(data.pain_frequency),
        "symptom_flags": extract_symptom_flags(data),
        "pain_location": data.pain_location # Add pain_location to processed data
    }
    return processed


def categorize_age(age: int) -> str:
    """Categorize age into relevant groups"""
    if age < 20:
        return "teens"
    elif age < 30:
        return "twenties"
    elif age < 40:
        return "thirties"
    elif age < 50:
        return "forties"
    else:
        return "fifties_plus"


def score_cycle_regularity(regularity: str) -> float:
    """Score cycle regularity (higher = more regular)"""
    scores = {
        "regular": 1.0,
        "irregular": 0.5,
        "very_irregular": 0.0
    }
    return scores.get(regularity, 0.5)


def score_pain_frequency(frequency: str) -> float:
    """Score pain frequency (higher = more frequent)"""
    scores = {
        "every_cycle": 1.0,
        "most_cycles": 0.8,
        "some_cycles": 0.4,
        "rarely": 0.1
    }
    return scores.get(frequency, 0.5)


def extract_symptom_flags(data: PatientData) -> Dict[str, bool]:
    """Extract boolean flags for key symptoms"""
    return {
        "heavy_bleeding": data.heavy_bleeding,
        "clotting": data.clotting,
        "pain_during_sex": data.pain_during_sex,
        "bowel_pain": data.bowel_pain,
        "nausea": data.nausea,
        "vomiting": data.vomiting,
        "diarrhea": data.diarrhea,
        "fatigue": data.fatigue,
        "headache": data.headache,
        "bloating": data.bloating,
        "unusual_discharge": data.unusual_discharge, # Added
        "fever": data.fever, # Added
        "chronic_pain": data.pain_frequency in ["every_cycle", "most_cycles"],
        "severe_pain": data.pain_severity >= 7,
        "pelvic_location": "pelvis" in data.pain_location,
        "back_location": "back" in data.pain_location,
        "abdomen_location": "lower_abdomen" in data.pain_location
    }


def score_condition(processed_patient_data: Dict[str, Any], condition_key: str) -> float:
    """Score a specific condition based on patient data and symptom mapping"""
    score = 0.0
    condition_map = SYMPTOM_CONDITION_MAPPING.get(condition_key, {})

    # Score based on key symptoms
    for symptom in condition_map.get("key_symptoms", []):
        if processed_patient_data["symptom_flags"].get(symptom, False):
            score += 1.0

    # Score based on pain characteristics (simplified for now)
    # This part can be expanded with more sophisticated matching
    if "pain_characteristics" in condition_map:
        pain_char = condition_map["pain_characteristics"]
        if "location" in pain_char and any(loc in processed_patient_data["pain_location"] for loc in pain_char["location"]):
            score += 0.5
        # Add more pain characteristic matching here (e.g., type, timing, severity)

    # Score based on age range
    if "age_range" in condition_map:
        if condition_key == "primary_dysmenorrhea" and processed_patient_data["age_category"] in ["teens", "twenties"]:
            score += 1.0
        elif condition_key == "endometriosis" and processed_patient_data["age_category"] in ["twenties", "thirties", "forties"]:
            score += 1.0
        elif condition_key == "fibroids" and processed_patient_data["age_category"] in ["thirties", "forties"]:
            score += 1.0
        elif condition_key == "pcos_related" and processed_patient_data["age_category"] in ["twenties", "thirties"]:
            score += 1.0
        elif condition_key == "adenomyosis" and processed_patient_data["age_category"] in ["thirties", "forties"]:
            score += 1.0

    # Score based on prior diagnoses and medical/family history
    if "prior_diagnoses" in processed_patient_data:
        for diagnosis in processed_patient_data["prior_diagnoses"]:
            if diagnosis.replace("_", "") in condition_key.replace("_", ""):
                score += 2.0 # Strong indicator

    if "medical_history" in processed_patient_data:
        for history_item in processed_patient_data["medical_history"]:
            if history_item == "fertility_issues" and condition_key == "endometriosis":
                score += 1.0

    if "family_history_reproductive" in processed_patient_data:
        for family_history_item in processed_patient_data["family_history_reproductive"]:
            if family_history_item.replace("_", "") in condition_key.replace("_", ""):
                score += 1.5 # Moderate indicator

    # Add specific symptom flags to score
    if processed_patient_data["symptom_flags"].get("heavy_bleeding") and condition_key in ["fibroids", "adenomyosis", "endometriosis"]:
        score += 1.0
    if processed_patient_data["symptom_flags"].get("severe_pain") and condition_key in ["endometriosis", "adenomyosis"]:
        score += 1.0
    if processed_patient_data["symptom_flags"].get("pelvic_location") and condition_key in ["endometriosis", "pid", "adenomyosis", "pcos_related"]:
        score += 0.5
    if processed_patient_data["symptom_flags"].get("pain_during_sex") and condition_key in ["endometriosis", "pid", "adenomyosis"]:
        score += 1.0
    if processed_patient_data["symptom_flags"].get("bowel_pain") and condition_key == "endometriosis":
        score += 1.0
    if processed_patient_data["symptom_flags"].get("nausea") and condition_key == "primary_dysmenorrhea":
        score += 0.5
    if processed_patient_data["symptom_flags"].get("bloating") and condition_key == "pcos_related":
        score += 0.5

    return score


def predict_pain_cause(patient_data: PatientData) -> PredictionResult:
    """Predicts the most likely cause of period pain"""
    processed_data = preprocess_patient_data(patient_data)
    
    # Add original history data to processed_data for score_condition to access
    processed_data["prior_diagnoses"] = patient_data.prior_diagnoses
    processed_data["medical_history"] = patient_data.medical_history
    processed_data["family_history_reproductive"] = patient_data.family_history_reproductive

    scores = {
        PainCause.PRIMARY_DYSMENORRHEA: score_condition(processed_data, "primary_dysmenorrhea"),
        PainCause.ENDOMETRIOSIS: score_condition(processed_data, "endometriosis"),
        PainCause.FIBROIDS: score_condition(processed_data, "fibroids"),
        PainCause.PCOS_RELATED: score_condition(processed_data, "pcos_related"),
        PainCause.PID: score_condition(processed_data, "pid"),
        PainCause.ADENOMYOSIS: score_condition(processed_data, "adenomyosis"),
    }

    # Determine the highest score
    max_score = 0.0
    predicted_cause = PainCause.UNKNOWN
    for cause, score in scores.items():
        if score > max_score:
            max_score = score
            predicted_cause = cause

    # Determine confidence level and clinical notes
    confidence = ConfidenceLevel.LOW
    clinical_notes = "Based on the provided symptoms, further medical evaluation is recommended."
    red_flags = []

    if predicted_cause != PainCause.UNKNOWN:
        # Simple confidence logic: if max_score is high, confidence is high
        if max_score >= 5.0: # Threshold can be tuned
            confidence = ConfidenceLevel.HIGH
            clinical_notes = f"The symptoms strongly suggest {predicted_cause.value.replace('_', ' ')}. Consider consulting a specialist for confirmation."
        elif max_score >= 2.0:
            confidence = ConfidenceLevel.MODERATE
            clinical_notes = f"The symptoms are consistent with {predicted_cause.value.replace('_', ' ')}. Further investigation may be beneficial."
        else:
            clinical_notes = f"While {predicted_cause.value.replace('_', ' ')} is a possibility, the evidence is not strong. Consider other potential causes or a general medical check-up."

    # Identify red flags
    if processed_data["symptom_flags"].get("unusual_discharge") or processed_data["symptom_flags"].get("fever"):
        red_flags.append("Signs of potential infection (e.g., PID). Seek immediate medical attention.")
    if patient_data.pain_severity >= 7 and patient_data.pain_duration_days > 7:
        red_flags.append("Prolonged severe pain. May indicate a serious underlying condition.")
    if patient_data.prior_diagnoses and "pid" in patient_data.prior_diagnoses:
        red_flags.append("History of PID. Recurrence or chronic issues possible.")

    return PredictionResult(
        predicted_cause=predicted_cause,
        confidence_level=confidence,
        clinical_notes=clinical_notes,
        red_flags=red_flags,
        scores=scores
    )


# Example Usage (for testing and demonstration)
if __name__ == "__main__":
    sample_patient_data = PatientData(
        age=25,
        sex_assigned_at_birth="female",
        height_cm=165.0,
        weight_kg=60.0,
        age_of_menarche=13,
        cycle_regularity="regular",
        period_duration_days=5,
        pain_frequency="every_cycle",
        pain_severity=8,
        pain_location=["lower_abdomen", "back", "pelvis"],
        pain_duration_days=7,
        heavy_bleeding=True,
        clotting=True,
        pain_during_sex=True,
        bowel_pain=True,
        nausea=False,
        vomiting=False,
        diarrhea=False,
        fatigue=True,
        headache=False,
        bloating=True,
        prior_diagnoses=[],
        medical_history=[],
        family_history_reproductive=["endometriosis"],
        unusual_discharge=False,
        fever=False,
        gender_identity="female",
        race_ethnicity="caucasian"
    )

    result = predict_pain_cause(sample_patient_data)
    print(json.dumps({
        "predicted_cause": result.predicted_cause.value,
        "confidence_level": result.confidence_level.value,
        "clinical_notes": result.clinical_notes,
        "red_flags": result.red_flags,
        "scores": {k.value: v for k, v in result.scores.items()}
    }, indent=4))

    # Another example for PID
    sample_patient_data_pid = PatientData(
        age=30,
        sex_assigned_at_birth="female",
        height_cm=160.0,
        weight_kg=65.0,
        age_of_menarche=12,
        cycle_regularity="irregular",
        period_duration_days=7,
        pain_frequency="some_cycles",
        pain_severity=7,
        pain_location=["pelvis", "lower_abdomen"],
        pain_duration_days=10,
        heavy_bleeding=False,
        clotting=False,
        pain_during_sex=True,
        bowel_pain=False,
        nausea=True,
        vomiting=False,
        diarrhea=False,
        fatigue=False,
        headache=False,
        bloating=False,
        prior_diagnoses=["pid"],
        medical_history=[],
        family_history_reproductive=[],
        unusual_discharge=True,
        fever=True,
        gender_identity="female",
        race_ethnicity="african_american"
    )

    result_pid = predict_pain_cause(sample_patient_data_pid)
    print(json.dumps({
        "predicted_cause": result_pid.predicted_cause.value,
        "confidence_level": result_pid.confidence_level.value,
        "clinical_notes": result_pid.clinical_notes,
        "red_flags": result_pid.red_flags,
        "scores": {k.value: v for k, v in result_pid.scores.items()}
    }, indent=4))




def validate_patient_data(data: PatientData):
    """Validates the input PatientData to ensure data integrity and correctness."""
    if not (10 <= data.age <= 100):
        raise ValueError("Age must be between 10 and 100.")
    if data.sex_assigned_at_birth not in ["female", "male", "intersex"]:
        raise ValueError("Sex assigned at birth must be 'female', 'male', or 'intersex'.")
    if not (50 <= data.height_cm <= 250):
        raise ValueError("Height must be between 50 and 250 cm.")
    if not (20 <= data.weight_kg <= 300):
        raise ValueError("Weight must be between 20 and 300 kg.")
    if not (8 <= data.age_of_menarche <= 20):
        raise ValueError("Age of menarche must be between 8 and 20.")
    if data.cycle_regularity not in ["regular", "irregular", "very_irregular"]:
        raise ValueError("Cycle regularity must be 'regular', 'irregular', or 'very_irregular'.")
    if not (1 <= data.period_duration_days <= 15):
        raise ValueError("Period duration must be between 1 and 15 days.")
    if data.pain_frequency not in ["every_cycle", "most_cycles", "some_cycles", "rarely"]:
        raise ValueError("Pain frequency must be 'every_cycle', 'most_cycles', 'some_cycles', or 'rarely'.")
    if not (1 <= data.pain_severity <= 10):
        raise ValueError("Pain severity must be between 1 and 10.")
    if not all(loc in ["lower_abdomen", "back", "thighs", "pelvis"] for loc in data.pain_location):
        raise ValueError("Invalid pain location provided.")
    if not (1 <= data.pain_duration_days <= 30):
        raise ValueError("Pain duration must be between 1 and 30 days.")

    # Check for valid boolean types
    for attr in ['heavy_bleeding', 'clotting', 'pain_during_sex', 'bowel_pain', 'nausea', 'vomiting', 'diarrhea', 'fatigue', 'headache', 'bloating', 'unusual_discharge', 'fever']:
        if not isinstance(getattr(data, attr), bool):
            raise ValueError(f"'{attr}' must be a boolean value.")

    # Check for valid list types for history
    for attr in ['prior_diagnoses', 'medical_history', 'family_history_reproductive']:
        if not isinstance(getattr(data, attr), list):
            raise ValueError(f"'{attr}' must be a list.")


# Modify predict_pain_cause to include validation
def predict_pain_cause_with_validation(patient_data: PatientData) -> PredictionResult:
    validate_patient_data(patient_data)
    return predict_pain_cause(patient_data)


# Update example usage to use the validated function
if __name__ == "__main__":
    sample_patient_data = PatientData(
        age=25,
        sex_assigned_at_birth="female",
        height_cm=165.0,
        weight_kg=60.0,
        age_of_menarche=13,
        cycle_regularity="regular",
        period_duration_days=5,
        pain_frequency="every_cycle",
        pain_severity=8,
        pain_location=["lower_abdomen", "back", "pelvis"],
        pain_duration_days=7,
        heavy_bleeding=True,
        clotting=True,
        pain_during_sex=True,
        bowel_pain=True,
        nausea=False,
        vomiting=False,
        diarrhea=False,
        fatigue=True,
        headache=False,
        bloating=True,
        unusual_discharge=False,
        fever=False,
        prior_diagnoses=[],
        medical_history=[],
        family_history_reproductive=["endometriosis"],
        gender_identity="female",
        race_ethnicity="caucasian"
    )

    result = predict_pain_cause_with_validation(sample_patient_data)
    print(json.dumps({
        "predicted_cause": result.predicted_cause.value,
        "confidence_level": result.confidence_level.value,
        "clinical_notes": result.clinical_notes,
        "red_flags": result.red_flags,
        "scores": {k.value: v for k, v in result.scores.items()}
    }, indent=4))

    # Another example for PID
    sample_patient_data_pid = PatientData(
        age=30,
        sex_assigned_at_birth="female",
        height_cm=160.0,
        weight_kg=65.0,
        age_of_menarche=12,
        cycle_regularity="irregular",
        period_duration_days=7,
        pain_frequency="some_cycles",
        pain_severity=7,
        pain_location=["pelvis", "lower_abdomen"],
        pain_duration_days=10,
        heavy_bleeding=False,
        clotting=False,
        pain_during_sex=True,
        bowel_pain=False,
        nausea=True,
        vomiting=False,
        diarrhea=False,
        fatigue=False,
        headache=False,
        bloating=False,
        unusual_discharge=True,
        fever=True,
        prior_diagnoses=["pid"],
        medical_history=[],
        family_history_reproductive=[],
        gender_identity="female",
        race_ethnicity="african_american"
    )

    result_pid = predict_pain_cause_with_validation(sample_patient_data_pid)
    print(json.dumps({
        "predicted_cause": result_pid.predicted_cause.value,
        "confidence_level": result_pid.confidence_level.value,
        "clinical_notes": result_pid.clinical_notes,
        "red_flags": result_pid.red_flags,
        "scores": {k.value: v for k, v in result_pid.scores.items()}
    }, indent=4))

    # Example of invalid data to test validation
    try:
        invalid_data = PatientData(
            age=5,
            sex_assigned_at_birth="female",
            height_cm=150.0,
            weight_kg=50.0,
            age_of_menarche=10,
            cycle_regularity="regular",
            period_duration_days=5,
            pain_frequency="every_cycle",
            pain_severity=5,
            pain_location=["lower_abdomen"],
            pain_duration_days=3,
            heavy_bleeding=False,
            clotting=False,
            pain_during_sex=False,
            bowel_pain=False,
            nausea=False,
            vomiting=False,
            diarrhea=False,
            fatigue=False,
            headache=False,
            bloating=False,
            unusual_discharge=False,
            fever=False,
            prior_diagnoses=[],
            medical_history=[],
            family_history_reproductive=[]
        )
        predict_pain_cause_with_validation(invalid_data)
    except ValueError as e:
        print(f"Validation Error: {e}")
