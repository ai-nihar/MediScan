def get_prevention_tips(disease_type, is_positive):
    """
    Returns a Python dictionary containing clinical prevention and management guidelines
    based on the disease type and screening status (positive/negative detection).
    """
    disclaimer = 'This is an AI screening tool. Always consult a qualified medical professional for diagnosis and treatment.'

    tips = {
        'pneumonia': {
            True: {
                'severity': 'Consult a doctor immediately',
                'immediate_steps': [
                    'Seek medical evaluation within 24 hours to obtain a professional diagnosis and prescription.',
                    'Monitor blood oxygen saturation (SpO2) levels regularly using a home pulse oximeter.',
                    'Control high fever and pain using over-the-counter antipyretics (e.g., acetaminophen) as directed by a physician.',
                    'Seek emergency medical care immediately if you experience severe chest pain, shortness of breath, or confusion.'
                ],
                'lifestyle': [
                    'Secure extensive bed rest and avoid physical exertion to allow your body to combat the infection.',
                    'Stay well-hydrated by consuming plenty of water, warm clear broths, or herbal teas.',
                    'Use a cool-mist humidifier or take warm, steamy showers to help loosen and clear lung secretions.',
                    'Avoid exposure to tobacco smoke, vaping, chemical fumes, and other respiratory irritants.'
                ],
                'medications': 'As prescribed by a doctor — typically antibiotics for bacterial pneumonia, or antiviral/supportive therapies for viral infections.',
                'follow_up': 'Schedule a repeat chest X-ray in 4-6 weeks to ensure the lung consolidation has fully resolved.'
            },
            False: {
                'severity': 'No active infection detected. Maintain healthy lung functions.',
                'immediate_steps': [
                    'Maintain strict hand hygiene by washing hands frequently with soap and water for at least 20 seconds.',
                    'Stay up to date with immunizations, including the annual influenza vaccine and pneumococcal vaccine if eligible.',
                    'Avoid close contact with individuals exhibiting respiratory symptoms like coughing, sneezing, or congestion.',
                    'Regularly clean and disinfect frequently touched household and workplace surfaces.'
                ],
                'lifestyle': [
                    'Abstain from smoking, vaping, or exposure to secondhand smoke to protect your lung cilia and immune cells.',
                    'Engage in regular cardiovascular exercise to promote optimal lung capacity and airway clearance.',
                    'Consume a balanced, nutrient-rich diet to support your immune system\'s response to pathogens.',
                    'Ensure adequate ventilation and clean air filters in indoor living and working environments.'
                ],
                'medications': 'No respiratory medications required. Do not take self-prescribed or residual antibiotics.',
                'follow_up': 'Schedule routine annual physical checkups with your primary care provider.'
            }
        },
        'retinopathy': {
            True: {
                'severity': 'Consult an ophthalmologist / retina specialist promptly',
                'immediate_steps': [
                    'Schedule an appointment with an ophthalmologist or retinal specialist for formal clinical staging.',
                    'Monitor blood glucose levels tightly and work with your endocrinologist to stabilize glycemic control.',
                    'Measure and regulate your blood pressure frequently, targeting a reading below 130/80 mmHg.',
                    'Seek immediate emergency eye care if you experience sudden vision loss, floating spots, or dark shadows.'
                ],
                'lifestyle': [
                    'Adopt a low-glycemic, heart-healthy diet to manage glucose, lipid profiles, and cardiovascular strain.',
                    'Engage in safe, low-impact exercise (avoid heavy lifting, straining, or valsalva maneuvers that stress retinal vessels).',
                    'Stop smoking completely to reduce microvascular damage and lower the risk of diabetic macular edema.',
                    'Wear UV-protective sunglasses outdoors to shield your eyes from solar radiation.'
                ],
                'medications': 'As prescribed by an ophthalmologist — typically includes intravitreal injections (anti-VEGF), retinal laser photocoagulation, or steroid implants.',
                'follow_up': 'Ophthalmology evaluations every 3-6 months, or as specifically directed by your eye care specialist.'
            },
            False: {
                'severity': 'No retinopathy detected. Maintain preventive diabetic eye care.',
                'immediate_steps': [
                    'Schedule a comprehensive dilated eye exam once a year, as early retinopathy is asymptomatic.',
                    'Have your HbA1c levels checked every 3 to 6 months, targeting a level below 7.0% as advised by your endocrinologist.',
                    'Maintain consistent daily monitoring of blood glucose levels to prevent microvascular damage.',
                    'Track blood pressure and cholesterol levels to protect retinal capillary walls.'
                ],
                'lifestyle': [
                    'Eat a diet rich in antioxidants, lutein, leafy green vegetables, and omega-3 fatty acids to support macular health.',
                    'Stay physically active with regular moderate exercise to support healthy system-wide vascular circulation.',
                    'Ensure proper task lighting when reading or working on digital screens to reduce eye strain.',
                    'Limit screen time and take periodic breaks using the 20-20-20 rule to rest your eyes.'
                ],
                'medications': 'Continue glucose-lowering medications (insulin or oral agents) exactly as prescribed by your doctor.',
                'follow_up': 'Routine annual dilated eye examination.'
            }
        },
        'skin_cancer': {
            True: {
                'severity': 'Urgent referral to a dermatologist / oncologist required',
                'immediate_steps': [
                    'Schedule an urgent consultation with a board-certified dermatologist for a clinical biopsy or excision.',
                    'Do not scratch, pick, shave, or manipulate the lesion to prevent irritation and ensure sample integrity.',
                    'Take a close-up photograph of the lesion next to a ruler to document size and monitor changes while waiting.',
                    'Check for any swollen lymph nodes in the neck, armpits, or groin near the lesion area.'
                ],
                'lifestyle': [
                    'Shield the lesion from sunlight entirely using physical covers (clothing, bandages) and hats.',
                    'Avoid outdoor activities during peak UV index hours (10:00 AM to 4:00 PM).',
                    'Conduct a full-body self-skin examination to catalog other moles and spots of concern.',
                    'Inform first-degree family members of the biopsy, as genetic factors increase melanoma risk.'
                ],
                'medications': 'Surgical excision is the first-line treatment. Targeted therapies, immunotherapies, or radiation are managed by oncology.',
                'follow_up': 'Skin oncology checkups every 3 to 6 months post-excision.'
            },
            False: {
                'severity': 'No malignancy detected. Continue routine skin protection.',
                'immediate_steps': [
                    'Perform a full-body skin self-check monthly using the ABCDE guidelines (Asymmetry, Border, Color, Diameter, Evolving).',
                    'Apply broad-spectrum sunscreen (SPF 30 or higher) daily to all exposed skin, even on cloudy days.',
                    'Reapply sunscreen every 2 hours when outdoors, or immediately after swimming or heavy sweating.',
                    'Schedule an annual professional skin exam with a board-certified dermatologist.'
                ],
                'lifestyle': [
                    'Wear tightly woven sun-protective clothing, a wide-brimmed hat, and UV-blocking sunglasses.',
                    'Avoid tanning beds, sun lamps, and deliberate outdoor tanning, which directly damage skin cell DNA.',
                    'Seek shade whenever possible, especially in midday sun.',
                    'Keep skin well-moisturized and inspect any new sores or spots that fail to heal.'
                ],
                'medications': 'No active medications needed. Apply high-quality protective sunblocks daily.',
                'follow_up': 'Annual screening exams by a dermatologist.'
            }
        }
    }

    # Retrieve specific guidelines based on disease type and status
    category = tips.get(disease_type, {})
    tip_dict = category.get(is_positive, {
        'severity': 'Consult a medical professional',
        'immediate_steps': ['Seek medical evaluation.'],
        'lifestyle': ['Maintain healthy habits.'],
        'medications': 'As advised by doctor.',
        'follow_up': 'As advised by doctor.'
    })
    
    # Add disclaimer
    tip_dict['disclaimer'] = disclaimer
    return tip_dict
