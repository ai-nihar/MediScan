from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Prediction
from .utils.predict import predict_disease
from .utils.prevention import get_prevention_tips

@login_required(login_url='/accounts/login/')
def upload_view(request):
    """
    Handle the diagnostic image upload page and kick off model inference.
    """
    if request.method == 'POST':
        disease_type = request.POST.get('disease_type')
        image_file = request.FILES.get('image')
        
        if not disease_type or not image_file:
            return render(request, 'predictor/upload.html', {
                'error': 'Please provide both a disease type and an image.'
            })
            
        # Save placeholder record to get image file path on disk
        prediction = Prediction(
            user=request.user,
            disease_type=disease_type,
            image=image_file,
            result='Processing',
            confidence=0.0,
            prevention_tips=''
        )
        prediction.save()
        
        # Run inference using the preprocessed saved image
        try:
            from .utils.predict import predictor
            
            # Get detailed prediction dictionary
            res = predictor.predict(prediction.image.path, disease_type)
            result = res['result']
            confidence = res['confidence']
            is_positive = res['is_positive']
            
            # Retrieve prevention tips dictionary based on classification status
            tips_dict = get_prevention_tips(disease_type, is_positive)
            
            # Format the dictionary into a clean, structured bullet-point layout
            formatted_tips = f"Severity Level: {tips_dict['severity']}\n\nImmediate Steps:\n"
            for step in tips_dict['immediate_steps']:
                formatted_tips += f"- {step}\n"
            formatted_tips += "\nLifestyle Recommendations:\n"
            for item in tips_dict['lifestyle']:
                formatted_tips += f"- {item}\n"
            formatted_tips += f"\nMedications:\n- {tips_dict['medications']}\n"
            formatted_tips += f"\nFollow-up:\n- {tips_dict['follow_up']}\n"
            formatted_tips += f"\nDisclaimer: {tips_dict['disclaimer']}"
            
            # Update prediction record
            prediction.result = result
            prediction.confidence = confidence
            prediction.prevention_tips = formatted_tips
            prediction.save()
            
            return redirect('predictor:result', pk=prediction.pk)
        except Exception as e:
            prediction.result = 'Error'
            prediction.prevention_tips = f"Error during model processing: {str(e)}"
            prediction.save()
            return redirect('predictor:result', pk=prediction.pk)

    return render(request, 'predictor/upload.html')

@login_required(login_url='/accounts/login/')
def result_view(request, pk):
    """
    Display the prediction result, confidence, and prevention tips.
    """
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    return render(request, 'predictor/result.html', {'prediction': prediction})
