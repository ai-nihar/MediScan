import requests
from bs4 import BeautifulSoup

def get_prevention_tips(disease_type, result):
    """
    Return clinical prevention and actionable health advice for the diagnosed condition.
    Includes scraping capability and high-quality, professional fallback guidelines.
    """
    
    # Clinical-grade fallback guidelines
    fallback_tips = {
        'pneumonia': {
            'Pneumonia': (
                "1. Clinical Consultation: Seek immediate medical evaluation from a physician.\n"
                "2. Medication Adherence: Take all prescribed antibiotics or antivirals exactly as directed, even if you start feeling better.\n"
                "3. Deep Breathing Exercises: Practice incentive spirometry or deep breathing to help clear lung congestion.\n"
                "4. Hydration & Rest: Consume warm fluids and secure extensive bed rest to assist your body in fighting the infection.\n"
                "5. Symptom Tracking: Monitor temperature, heart rate, and oxygen saturation levels closely. Seek emergency care for shortness of breath."
            ),
            'Normal': (
                "1. Immunization: Keep up-to-date with the pneumococcal vaccine and the annual flu shot.\n"
                "2. Hand Hygiene: Wash hands frequently with soap and water to prevent contracting respiratory pathogens.\n"
                "3. Respiratory Etiquette: Avoid close contact with individuals exhibiting cold or flu-like symptoms.\n"
                "4. Pulmonary Health: Avoid smoking and secondhand smoke, which damage the lung cilia and increase infection risk."
            )
        },
        'retinopathy': {
            'Diabetic Retinopathy': (
                "1. Glycemic Control: Work with your endocrinologist to stabilize blood glucose levels (HbA1c target usually < 7%).\n"
                "2. Retinal Specialist Care: Schedule a prompt appointment with an ophthalmologist for staging and to discuss therapies like photocoagulation or anti-VEGF injections.\n"
                "3. Cardiovascular Management: Keep blood pressure (< 130/80 mmHg) and lipid levels under control to reduce macular edema.\n"
                "4. Vision Monitoring: Perform regular self-checks using an Amsler grid to catch sudden changes or distortions in central vision.\n"
                "5. Physical Activity: Continue safe, low-impact exercise as recommended by your physician."
            ),
            'No Diabetic Retinopathy': (
                "1. Annual Dilated Eye Exams: Schedule comprehensive eye examinations yearly, as early retinopathy is asymptomatic.\n"
                "2. Glucose Management: Maintain strict glycemic control to prevent microvascular damage to the retina.\n"
                "3. Vascular Health: Monitor blood pressure and cholesterol levels regularly.\n"
                "4. Balanced Nutrition: Consume antioxidant-rich foods, leafy greens, and omega-3 fatty acids to support overall retinal health."
            )
        },
        'skin_cancer': {
            'Malignant Melanoma': (
                "1. Urgent Dermatological Referral: Schedule an immediate consultation with a surgical dermatologist or oncologist for staging and excision planning.\n"
                "2. Avoid Lesion Manipulation: Do not scratch, pick, or attempt to self-treat the suspicious lesion.\n"
                "3. Full UV Shielding: Wear broad-spectrum SPF 50+ sunscreen, a wide-brimmed hat, and UV-blocking clothing outdoors.\n"
                "4. Full Body Mapping: Have a professional complete a full-body mole mapping session to examine other atypical moles.\n"
                "5. Family Awareness: Inform first-degree relatives, as genetic predisposition plays a significant role in melanoma risk."
            ),
            'Benign Lesion': (
                "1. ABCDE Monitoring: Conduct self-checks monthly using the ABCDE guidelines (Asymmetry, Border irregularity, Color variation, Diameter > 6mm, Evolving shape/size).\n"
                "2. Sun Protection: Apply broad-spectrum SPF 30+ daily to all exposed skin, reapplying every 2 hours during direct sun exposure.\n"
                "3. Avoid Artificial UV: Do not use tanning beds or booths, which dramatically raise skin cancer risk.\n"
                "4. Preventive Screenings: Visit a dermatologist annually for a professional skin examination."
            )
        }
    }

    # Attempt to fetch dynamic tips from Wikipedia or WebMD (as a scraping demonstration)
    # If it fails or times out, it falls back to the high-quality local guidelines.
    try:
        if disease_type == 'pneumonia' and result == 'Pneumonia':
            url = "https://en.wikipedia.org/wiki/Pneumonia"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Grab the first paragraph of the prevention section
                prevention_header = soup.find(id="Prevention")
                if prevention_header:
                    p_tag = prevention_header.find_parent().find_next_sibling('p')
                    if p_tag and len(p_tag.text.strip()) > 50:
                        scraped_tip = "Scraped Wikipedia Guideline:\n" + p_tag.text.strip()[:300] + "...\n\n"
                        return scraped_tip + fallback_tips['pneumonia']['Pneumonia']
    except Exception as e:
        # Ignore scraping errors and rely on clinical fallbacks
        pass

    category_tips = fallback_tips.get(disease_type, {})
    return category_tips.get(result, "Consult a medical professional for diagnosis and advice.")
