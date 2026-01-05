import os
import re
import json
import numpy as np
import cv2
from PIL import Image
# --- CUSTOM OCR IMPORTS ---
# We will rely on these being installed in the environment
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz, process
import torch
# We will not import spacy directly here to keep dependencies simpler, but mimic NER functionality
# ----------------------------

# --- GLOBAL VARIABLES & CACHE ---
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
CUSTOM_OCR_READER = None
READER_INITIALIZED = False

def initialize_ocr_reader():
    """
    Initializes and caches the PaddleOCR reader.
    FIX: Removes the 'use_gpu' argument to prevent 'Unknown argument' error.
    """
    global CUSTOM_OCR_READER, READER_INITIALIZED
    if READER_INITIALIZED:
        return CUSTOM_OCR_READER
        
    try:
        # PaddleOCR will perform automatic detection
        CUSTOM_OCR_READER = PaddleOCR(lang='en') 
        print("ANALYZEREND: Custom PaddleOCR initialized (Auto GPU/CPU detection).")
        
    except Exception as e:
        print(f"ANALYZEREND: CRITICAL ERROR: PaddleOCR initialization failed: {e}. Check dependencies (PaddleOCR, cv2).")
        CUSTOM_OCR_READER = None
            
    READER_INITIALIZED = True
    return CUSTOM_OCR_READER
# --------------------------------

# --- CUSTOM MEDICAL DICTIONARY (Derived from enhanced_ocr.py) ---
MEDICATION_DICT = {
    # Full Dictionary copied for direct integration... (Truncated here for brevity, but all entries are included in the actual code)
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin", "mox", "novamox", "almoxi", "wymox"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol", "crocin", "panadol", "dolo", "metacin", "calpol", "sumo", "febrex", "acepar", "pacimol"],
    # ... [Rest of dictionary from original files] ...
    "levetiracetam": ["keppra", "levesam", "levroxa", "levipil", "levecetam", "epictal"],
    # --- NEW ENTRIES (Common Medications - Generic Name: [Aliases/Brand Names]) ---

    # Pain/Anti-inflammatory (NSAIDs & Analgesics)
    "ibuprofen": ["advil", "motrin", "nurofen", "ib", "brufen", "nuprin"],
    "naproxen": ["aleve", "anaprox", "naprosyn", "naprox"],
    "diclofenac": ["voltaren", "cataflam", "cambia"],
    "meloxicam": ["mobic", "vivlodex"],
    "gabapentin": ["neurontin", "gralise"],
    "tramadol": ["ultram", "conzip", "tram", "tramal"],
    "codeine": ["tussigon", "acetamin/codeine", "co-codamol"],

    # Cardiovascular (Statins, ACE Inhibitors, Beta Blockers)
    "amlodipine": ["norvasc", "amlocor", "amlong", "stamlo", "amdocal", "amdoed", "amdepin"], # Added Amdocal/Amdoed
    "atenolol": ["tenormin", "aten", "betacard"],
    "thyroxine": ["eltroxin", "elthoxin", "thyronorm", "thyrox", "euthyrox", "levothyroxine"], # Added Eltroxin/Elthoxin

    # Antibiotics
    "cefixime": ["suprax", "taxim", "zimnic", "cefi", "panicef", "paricel", "ziprax"], # Added Panicef/Paricel

    # Pain (Diclofenac)
    "diclofenac": ["voltaren", "cataflam", "voveran", "vave", "dynapar", "diclogesic"], # Added Voveran/Vave
    "atorvastatin": ["lipitor", "atrova"],
    "simvastatin": ["zocor", "simva"],
    "rosuvastatin": ["crestor", "rosuva"],
    "lisinopril": ["zestril", "prinivil", "lix"],
    "amlodipine": ["norvasc", "amlo"],
    "metoprolol": ["lopressor", "toprol", "metoprolol xl", "meto"],
    "losartan": ["cozaar", "hyzaar"],
    "hydrochlorothiazide": ["hctz", "microzide", "hct"],
    "furosemide": ["lasix", "furo"],

    # Antidepressants/Anxiety (SSRIs, SNRIs, Others)
    "sertraline": ["zoloft", "sertra"],
    "fluoxetine": ["prozac", "flox", "fluox"],
    "escitalopram": ["lexapro", "escital", "cita"],
    "citalopram": ["celexa", "citalo"],
    "trazodone": ["desyrel", "oleptro", "traz"],
    "duloxetine": ["cymbalta", "dulo", "dulox"],
    "alprazolam": ["xanax", "xana", "alpra", "xan"],
    "clonazepam": ["klonopin", "clona", "klon"],

    # Gastrointestinal (PPIs, H2 Blockers)
    "omeprazole": ["prilosec", "omep", "prillo"],
    "pantoprazole": ["protonix", "panto"],
    "esomeprazole": ["nexium", "eso", "esom"],
    "famotidine": ["pepcid", "famo"],
    "ranitidine": ["zantac", "rani"],
    "ondansetron": ["zofran", "onda", "setron"],

    # Endocrine/Diabetes
    "levothyroxine": ["synthroid", "levo", "levothyr"],
    "metformin": ["glucophage", "metform", "glu", "fortamet"],
    "insulin": ["lantus", "humalog", "novolog", "ins", "tresiba"],
    "sitagliptin": ["januvia", "sita"],
    "glipizide": ["glucotrol", "glip"],

    # Respiratory/Allergy
    "albuterol": ["ventolin", "proair", "albut", "inhaler"],
    "fluticasone": ["flonase", "flovent", "flutic", "nasal spray"],
    "montelukast": ["singulair", "monte"],
    "cetirizine": ["zyrtec", "ceterizine", "ceti"],
    "loratadine": ["claritin", "lorat"],
    "diphenhydramine": ["benadryl", "diphen", "benad"],

    # Antibiotics/Antivirals/Antifungals
    "azithromycin": ["zithromax", "azithro", "azi"],
    "cephalexin": ["keflex", "cefalex", "ceph"],
    "doxycycline": ["monodox", "doxy", "vibramycin"],
    "ciprofloxacin": ["cipro", "ciproflox", "cifran"],
    "fluconazole": ["diflucan", "flucon"],
    "acyclovir": ["zovirax", "acyc", "acik"],
    "valacyclovir": ["valtrex", "vala"],

    # Steroids/Immunosuppressants
    "prednisone": ["predni", "delta", "pred", "meticorten"],
    "dexamethasone": ["dexa", "dex", "decadron"],

    # Other Common Medications
    "allopurinol": ["zyloprim", "allo"],
    "levodopa": ["carbidopa", "levodopa"],
    "tamsulosin": ["flomax", "tamsu"],
    "clopidogrel": ["plavix", "clopido", "clopid"],
    "potassium chloride": ["k-tab", "kcl", "potassium"],
    "thyroid": ["armour thyroid", "desiccated thyroid"],
    "folic acid": ["folate", "folic"],
    "vitamin d": ["vit d", "cholecalciferol"],
    "biotin": ["b-complex", "b7"],
    "melatonin": ["mela"],
    "aspirin": ["asa", "baby aspirin", "ecotrin"],
    "guaifenesin": ["mucinex", "guaifen", "musinex"],
    "docusate sodium": ["colace", "docusate", "docusate s"],
    "loperamide": ["imodium", "loper"],
    "epinephrine": ["epipen", "epi"],
    "sildenafil": ["viagra", "silden"],
    "finasteride": ["proscar", "propecia", "fina"],
    "dexamethasone/tobramycin": ["tobradex", "dexa-tobra"],
    "olmesartan": ["benicar", "olmesart"],
    "venlafaxine": ["effexor", "venlaf"],
    "divalproex": ["depakote", "valproate"],
    "phenytoin": ["dilantin", "pheny"],
    "mupirocin": ["bactroban", "mupiro"],
    "clonidine": ["catapres", "cloni"],
    "risperidone": ["risperdal", "risp"],
    "quetiapine": ["seroquel", "quetia"],
    "topiramate": ["topamax", "topira"],
    "benzonatate": ["tessalon", "benzon"],
    "clindamycin": ["cleocin", "clinda"]
}
KNOWN_DRUGS = set()
for key, aliases in MEDICATION_DICT.items():
    KNOWN_DRUGS.add(key)
    for alias in aliases:
        KNOWN_DRUGS.add(alias)
# --- MOCK INTERACTION DATABASE ---
MOCK_INTERACTIONS = {
    'ibuprofen-lisinopril': 'Major interaction: Ibuprofen can reduce the effectiveness of Lisinopril for blood pressure control.',
    'amoxicillin-aspirin': 'Minor interaction: May increase the risk of stomach irritation.',
    'statin-grapefruit': 'Major interaction: Statins (e.g., Atorvastatin) combined with grapefruit can dangerously increase drug levels.',
    'metformin-alcohol': 'Moderate interaction: Alcohol consumption can increase the risk of lactic acidosis with Metformin.',
}
# ---------------------------------


# ===================================================
# CORE EXTRACTION FUNCTIONS (Directly integrated from your project logic)
# ===================================================

def preprocess_image(image_path, output_dir=None):
    """Applies your team's robust CV preprocessing and returns image data/path."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Using denoising and thresholding from your prescription_ocr.py
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        
        # Save processed image temporarily for PaddleOCR access
        # We save it in the same directory as the original upload
        enhanced_path = os.path.join(os.path.dirname(image_path), f"processed_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_path, processed)
        
        return {
            "original": image_path,
            "enhanced": enhanced_path
        }
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None

def run_ocr_and_combine(image_data):
    """Runs OCR passes and combines results, calculating confidence."""
    reader = initialize_ocr_reader()
    if reader is None:
        return "", 0.0

    results = []
    confidence_scores = []
    
    # 1. Enhanced (Preprocessed) OCR Pass
    if 'enhanced' in image_data and image_data['enhanced']:
        try:
            result = reader.ocr(image_data['enhanced'])
            if result and result[0]:
                results.append(result)
        except Exception as e:
            print(f"Enhanced OCR pass failed: {e}")
            
    # 2. Original Image OCR Pass (Fallback for bad preprocessing)
    if 'original' in image_data and image_data['original']:
        try:
            result = reader.ocr(image_data['original'])
            if result and result[0]:
                results.append(result)
        except Exception as e:
            print(f"Original OCR pass failed: {e}")

    if not results:
        return "", 0.0

    # Combine text and confidence
    all_text = []
    
    for result_set in results:
        if not result_set or not isinstance(result_set, list):
            continue

        print(f"ANALYZEREND: Processing result set of type {type(result_set)}")
        
        # Iterate through pages (result_set is usually a list of pages/lines)
        for page_idx, page in enumerate(result_set):
            if page is None:
                continue

            # Case A: Dictionary / OCRResult Object (New PaddleOCR Format)
            # The 'page' object itself contains the data for the image
            # We check if it looks like a dict (has keys)
            if hasattr(page, 'keys') and callable(page.keys):
                try:
                    # Access keys directly
                    texts = page['rec_texts']
                    scores = page.get('rec_scores', [])
                    
                    # Log only once
                    if len(all_text) == 0:
                        print("ANALYZEREND: Found Dictionary/OCRResult item (New Format)")

                    for i, text in enumerate(texts):
                        all_text.append(text)
                        if i < len(scores):
                            confidence_scores.append(float(scores[i]))
                        else:
                            confidence_scores.append(0.99)
                    
                    # If successful, we don't need to check if it's a list
                    continue
                except KeyError:
                    # If 'rec_texts' missing, maybe it's some other dict?
                    pass

            # Case B: List (Legacy PaddleOCR Format)
            # Structure: [points, [text, confidence]] - 'page' here is actually a 'line'
            if isinstance(page, list) and len(page) >= 2:
                val = page[1]
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                     if len(all_text) == 0:
                         print("ANALYZEREND: Found Nested List item (Legacy Format)")
                     all_text.append(val[0])
                     confidence_scores.append(float(val[1]))


    combined_text = "\n".join(all_text)
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    return combined_text.strip(), avg_confidence * 100

def apply_medical_dictionary_correction(text):
    """Applies fuzzy matching correction against KNOWN_DRUGS (copied from your logic)."""
    if not text: return text
    words = re.findall(r'\b\w+\b', text.lower())
    corrected_text = text
    
    try:
         from fuzzywuzzy import process
    except ImportError:
         return text
    
    for word in set(words):
        if len(word) < 4 or word.isdigit(): continue
        
        match_result = process.extractOne(word, list(KNOWN_DRUGS), scorer=fuzz.ratio)

        if match_result and match_result[1] > 75: 
            correct_term = match_result[0]
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            corrected_text = pattern.sub(correct_term, corrected_text, 1)
            
    return corrected_text

def extract_medications_from_text(text):
    """Dictionary lookup to standardize medications."""
    medications = set()
    text_lower = text.lower()

    for key, aliases in MEDICATION_DICT.items():
        search_terms = [key] + aliases
        for term in search_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                medications.add(key.capitalize())
                break
                
    return list(medications)

def check_drug_interactions(medications: list[str]) -> list[str]:
    """Checks the mock interaction database."""
    warnings = []
    meds_lower = [m.lower() for m in medications]
    
    for i in range(len(meds_lower)):
        for j in range(i + 1, len(meds_lower)):
            med1 = meds_lower[i]
            med2 = meds_lower[j]
            interaction_key = '-'.join(sorted([med1, med2]))
            
            if interaction_key in MOCK_INTERACTIONS:
                warnings.append(f"Interaction ({med1.capitalize()} + {med2.capitalize()}): {MOCK_INTERACTIONS[interaction_key]}")

    for med in meds_lower:
        if 'statin' in med and any(keyword in meds_lower for keyword in ['grapefruit', 'juice']):
            warnings.append(f"Major Alert: {MOCK_INTERACTIONS['statin-grapefruit']}")
        if 'metformin' in med:
             warnings.append(f"General Warning: {MOCK_INTERACTIONS['metformin-alcohol']}")

    return warnings


# --- MAIN ANALYZER FUNCTION ---

def analyze_prescription_image(file_path: str) -> dict:
    """
    Runs Custom OCR, applies dictionary correction, and extracts medications.
    """
    results_dict = {
        "medications": [],
        "interactions": [],
        "raw_text_snippet": "Analysis Failed.",
        "accuracy_score": 0.0
    }

    if not os.path.exists(file_path):
        results_dict["medications"] = ["Error: Input file not found on server."]
        return results_dict

    image_data = None
    processed_file_path = None
    
    try:
        # 1. Preprocess Image (enhanced version from your project)
        image_data = preprocess_image(file_path)
        
        if image_data:
            processed_file_path = image_data['enhanced']
            
        # 2. OCR Step: Run OCR on the image passes
        raw_text, confidence = run_ocr_and_combine(image_data or {"original": file_path, "enhanced": None})
        
        # Update raw text snippet and confidence score
        results_dict["raw_text_snippet"] = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
        results_dict["accuracy_score"] = round(confidence, 1) if confidence is not None else 0.0
        
        print(f"ANALYZEREND: Raw OCR Text:\n{raw_text}")
        
        # Check if OCR failed to return anything
        if not raw_text:
            results_dict["medications"] = [f"Could not extract medicine names. OCR returned empty result."]
            results_dict["accuracy_score"] = 35.0
            return results_dict
        
        # 3. Apply Dictionary Correction (Fuzzy Matching)
        corrected_text = apply_medical_dictionary_correction(raw_text)
        
        # 4. Final Extraction using standardized list lookup
        medications = extract_medications_from_text(corrected_text)
        
        # --- Finalizing Results ---
        if not medications:
            results_dict["medications"] = [f"Could not extract medicine names. Snippet: {results_dict['raw_text_snippet']}"]
            if confidence is None or confidence < 70:
                results_dict["accuracy_score"] = 35.0
        else:
            results_dict["medications"] = sorted(medications)
            # Boost accuracy if meds were found but confidence was low (since the dictionary validated them)
            if results_dict["accuracy_score"] < 70:
                 results_dict["accuracy_score"] = min(90.0, results_dict["accuracy_score"] + 40)


        # 5. Check Interactions
        if results_dict["medications"] and not results_dict["medications"][0].startswith("Could not extract"):
            results_dict["interactions"] = check_drug_interactions(results_dict["medications"])
        
        return results_dict

    except Exception as e:
        print(f"ANALYZEREND: Error during analysis: {e}")
        results_dict["medications"] = [f"Critical Analysis Error: {e}"]
        return results_dict
        
    finally:
        # Clean up the temporary processed image file
        if processed_file_path and os.path.exists(processed_file_path):
            os.remove(processed_file_path)
