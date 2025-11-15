"""
System Simulation: User Identity and Product Recommendation System

This script implements the complete system flow as described in the assignment:
1. Facial Recognition Model (authentication checkpoint)
2. Product Recommendation Model (if face recognized)
3. Voice Validation Model (final approval)
4. Display Predicted Product (if voice validated)

The system runs in INTERACTIVE MODE by default, prompting the user for:
- Image path for facial recognition
- Customer data for product recommendation (purchase amount, rating, etc.)
- Audio path for voice validation
Then displays the predicted product.

The system also simulates unauthorized attempts to demonstrate security.

Usage:
    # Interactive mode (prompts for all inputs):
    python scripts/system_simulation.py
"""

import sys
import argparse
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from predict_face import predict_face
from predict_voice import predict_voice
from predict_product import predict_product


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def simulate_full_transaction(image_path, audio_path, customer_data=None, verified_person=None, face_confidence=None):
    """
    Simulate a full transaction flow:
    1. Facial Recognition → 2. Product Recommendation → 3. Voice Validation → 4. Display Product
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the facial image for recognition
    audio_path : str or Path
        Path to the audio file for voice validation
    customer_data : dict, optional
        Customer data for product recommendation. If None, uses default values.
    verified_person : str, optional
        If provided, skips facial recognition and uses this person (already verified)
    face_confidence : float, optional
        Confidence from previous facial recognition (if verified_person is provided)
    
    Returns:
    --------
    bool : True if transaction successful, False otherwise
    """
    print_header("SYSTEM SIMULATION: FULL TRANSACTION FLOW")
    
    # Initialize person variable to ensure it's defined
    person = None
    face_conf = 0.0
    
    # Define the list of trained/authorized members
    AUTHORIZED_MEMBERS = ['Erneste', 'Thierry', 'Idara', 'Rodas']
    
    # Step 1: Facial Recognition (skip if already verified)
    if verified_person is not None:
        # Person already verified in interactive mode
        person = verified_person
        face_conf = face_confidence if face_confidence is not None else 1.0
        print_section("STEP 1: FACIAL RECOGNITION (ALREADY VERIFIED)")
        print(f"Processing image: {image_path}")
        print("\n" + "-" * 70)
        print("FACIAL RECOGNITION RESULT:")
        print("-" * 70)
        print(f"Person: {person}")
        print(f"Confidence: {face_conf:.2%}")
        print(f"Status: SUCCESS (Pre-verified)")
        print(f"Decision: CONTINUE - Person is authorized")
        print("-" * 70)
        print(f"\nFACIAL RECOGNITION SUCCESSFUL (Pre-verified)")
        print(f"Identified as: {person}")
        print(f"Confidence: {face_conf:.2%}")
        print(f"Authorization: VERIFIED - Person is in authorized list")
        print(f"Proceeding to product recommendation...")
    else:
        # Run facial recognition
        print_section("STEP 1: FACIAL RECOGNITION")
        print(f"Processing image: {image_path}")
        
        try:
            # Call facial recognition
            person, face_conf = predict_face(image_path)
        
            # Always display the facial recognition result first
            print("\n" + "-" * 70)
            print("FACIAL RECOGNITION RESULT:")
            print("-" * 70)
            if person is None or person == "":
                print(f"Person: None (Not Recognized)")
                print(f"Confidence: {face_conf:.2%}")
                print(f"Status: FAILED")
                print(f"Decision: STOP - Person not found in training data")
            else:
                print(f"Person: {person}")
                print(f"Confidence: {face_conf:.2%}")
                # Check if person is in authorized list
                if person in AUTHORIZED_MEMBERS:
                    print(f"Status: SUCCESS")
                    print(f"Decision: CONTINUE - Person is authorized")
                else:
                    print(f"Status: FAILED")
                    print(f"Decision: STOP - Person not in authorized list")
            print("-" * 70)
            
            # CRITICAL CHECK 1: If person is None or empty, STOP immediately
            if person is None or person == "":
                print("\n" + "=" * 70)
                print("ACCESS DENIED: Facial recognition failed")
                print("=" * 70)
                print(f"Reason: Person not recognized")
                print(f"Confidence: {face_conf:.2%}")
                print(f"The person in this image was not found in the training data.")
                print(f"Authorized members: {', '.join(AUTHORIZED_MEMBERS)}")
                print("\n" + "=" * 70)
                print("SYSTEM SECURITY: Access denied at facial recognition stage")
                print("Product recommendation will NOT be executed.")
                print("=" * 70)
                return False  # STOP HERE - Do not continue
            
            # CRITICAL CHECK 2: Verify person is one of the authorized members
            if person not in AUTHORIZED_MEMBERS:
                print("\n" + "=" * 70)
                print("ACCESS DENIED: Person not authorized")
                print("=" * 70)
                print(f"Reason: '{person}' is not in the authorized members list")
                print(f"Confidence: {face_conf:.2%}")
                print(f"Authorized members: {', '.join(AUTHORIZED_MEMBERS)}")
                print(f"The system only allows access to trained members.")
                print("\n" + "=" * 70)
                print("SYSTEM SECURITY: Access denied at facial recognition stage")
                print("Product recommendation will NOT be executed.")
                print("=" * 70)
                return False  # STOP HERE - Do not continue
            
            # Only reach here if person is NOT None AND is in AUTHORIZED_MEMBERS
            print(f"\nFACIAL RECOGNITION SUCCESSFUL")
            print(f"Identified as: {person}")
            print(f"Confidence: {face_conf:.2%}")
            print(f"Authorization: VERIFIED - Person is in authorized list")
            print(f"Proceeding to product recommendation...")
            
        except Exception as e:
            print("\n" + "-" * 70)
            print("FACIAL RECOGNITION RESULT:")
            print("-" * 70)
            print(f"Person: None (Error occurred)")
            print(f"Confidence: 0.00%")
            print(f"Status: ERROR")
            print("-" * 70)
            print("\n" + "=" * 70)
            print("ACCESS DENIED: Error during facial recognition")
            print("=" * 70)
            print(f"Error: {str(e)}")
            print("\n" + "=" * 70)
            print("SYSTEM SECURITY: Access denied at facial recognition stage")
            print("Product recommendation will NOT be executed.")
            print("=" * 70)
            return False  # STOP HERE - Do not continue
    
    # Step 2: Product Recommendation (only if face recognized)
    # This step only executes if facial recognition was successful AND person is authorized
    # Final security check before proceeding
    AUTHORIZED_MEMBERS = ['Erneste', 'Thierry', 'Idara', 'Rodas']
    
    if person is None or person == "":
        print("\n" + "=" * 70)
        print("SECURITY CHECK: Person is None - Stopping execution")
        print("=" * 70)
        return False
    
    if person not in AUTHORIZED_MEMBERS:
        print("\n" + "=" * 70)
        print("SECURITY CHECK: Person not authorized - Stopping execution")
        print("=" * 70)
        print(f"Person '{person}' is not in the authorized members list.")
        print(f"Authorized members: {', '.join(AUTHORIZED_MEMBERS)}")
        print("=" * 70)
        return False
    
    print_section("STEP 2: PRODUCT RECOMMENDATION")
    print("Running product recommendation model...")
    print(f"(This step only runs if facial recognition was successful)")
    print(f"Security Check: Person '{person}' is authorized and recognized")
    
    # Use default customer data if not provided
    if customer_data is None:
        customer_data = {
            'purchase_amount': 300.0,
            'customer_rating': 4.5,
            'engagement_score': 75.0,
            'purchase_interest_score': 3.5,
            'social_media_platform': 'Facebook',
            'review_sentiment': 'Positive',
            'purchase_date': '2024-01-15'
        }
        print("Using default customer transaction data")
    
    # Initialize product variables (will be set after prediction)
    product = None
    product_confidence = 0.0
    
    try:
        product, product_confidence = predict_product(customer_data)
        
        if product is None:
            print("\nWARNING: Could not generate product recommendation")
            print(f"Confidence: {product_confidence:.2%}")
            # Continue to voice validation even if product prediction fails
        else:
            print(f"\nPRODUCT RECOMMENDATION GENERATED")
            print(f"Status: Product recommendation successfully generated")
            print(f"Confidence: {product_confidence:.2%}")
            print(f"\nIMPORTANT: Product name will be displayed ONLY after:")
            print(f"1. Voice validation passes")
            print(f"2. Voice matches face identity")
            print(f"Currently: Product is generated but name is NOT displayed yet")
        
    except Exception as e:
        print(f"\nWARNING: Error during product recommendation")
        print(f"Error: {str(e)}")
        product = None
        product_confidence = 0.0
        # Continue to voice validation
    
    # Step 3: Voice Validation (final approval)
    print_section("STEP 3: VOICE VALIDATION")
    print(f"Processing audio: {audio_path}")
    
    try:
        voice_person, voice_confidence = predict_voice(audio_path)
        
        # Always display the voice validation result first
        print("\n" + "-" * 70)
        print("VOICE VALIDATION RESULT:")
        print("-" * 70)
        if voice_person is None:
            print(f"Person: None (Not Recognized)")
            print(f"Confidence: {voice_confidence:.2%}")
            print(f"Status: FAILED")
        else:
            print(f"Person: {voice_person}")
            print(f"Confidence: {voice_confidence:.2%}")
            print(f"Status: SUCCESS")
        print("-" * 70)
        
        # Now check if we should continue or stop
        if voice_person is None:
            print("\nACCESS DENIED: Voice validation failed")
            print(f"Reason: Voice not recognized (confidence: {voice_confidence:.2%})")
            print("\n" + "=" * 70)
            return False
        
        # CRITICAL SECURITY CHECK: Verify voice matches face
        # Normalize strings for robust comparison (strip whitespace)
        face_normalized = str(person).strip() if person else ""
        voice_normalized = str(voice_person).strip() if voice_person else ""
        
        # Check if voice matches face - if NOT, STOP immediately and DO NOT display product
        voice_matches_face = (voice_normalized == face_normalized)
        
        if not voice_matches_face:
            print(f"\nACCESS DENIED: Voice does not match face")
            print(f"Face identified: {person}")
            print(f"Voice identified: {voice_person}")
            print(f"Voice confidence: {voice_confidence:.2%}")
            print(f"Match status: MISMATCH")
            print(f"\n" + "=" * 70)
            print("SYSTEM SECURITY: Identity mismatch detected")
            print("Access denied - voice and face must match to display product.")
            print("=" * 70)
            return False  # CRITICAL: Stop here, do NOT display product
        
        # Only reach here if voice_person == person (identity match confirmed)
        # This is the ONLY path where product can be displayed (per flowchart)
        print(f"\nVOICE VALIDATION SUCCESSFUL")
        print(f"Identified as: {voice_person}")
        print(f"Confidence: {voice_confidence:.2%}")
        print(f"Voice matches face identity: {person} == {voice_person}")
        print(f"Identity match confirmed - proceeding to display product")
        
        # FINAL SECURITY CHECK: One more verification before displaying product
        if voice_person != person or voice_normalized != face_normalized:
            print(f"\n" + "=" * 70)
            print("SECURITY ERROR: Final identity verification failed")
            print("=" * 70)
            print("Product recommendation will NOT be displayed.")
            print("Security check: Voice and face do not match.")
            print("=" * 70)
            return False
        
        # ============================================================
        # ONLY REACH HERE IF: Voice validation passed AND voice matches face
        # This is the ONLY place where product can be displayed (per flowchart)
        # ============================================================
        print_section("STEP 4: DISPLAY PREDICTED PRODUCT")
        print_header("TRANSACTION APPROVED - PRODUCT RECOMMENDATION")
        
        print(f"\nUser: {person}")
        print(f"Voice Verified: {voice_person}")
        print(f"Identity Match: CONFIRMED ({person} == {voice_person})")
        print(f"Security Check: PASSED - Voice matches face")
        print(f"\nAll authentication checks passed!")
        print(f"Face recognized: {person}")
        print(f"Voice verified: {voice_person}")
        print(f"Identity match confirmed: {person} == {voice_person}")
        
        # NOW display the product (only after all checks passed)
        if product is not None:
            print(f"\n" + "=" * 70)
            print("RECOMMENDED PRODUCT (NOW DISPLAYED):")
            print("=" * 70)
            print(f"Product: {product}")
            print(f"Confidence: {product_confidence:.2%}")
            print("=" * 70)
        else:
            print(f"\nWARNING: No product recommendation available")
            print(f"Product prediction failed or was not generated")
        
        print(f"\nTransaction approved! Product displayed successfully.")
        print("\n" + "=" * 70)
        
        return True
        
    except Exception as e:
        print("\n" + "-" * 70)
        print("VOICE VALIDATION RESULT:")
        print("-" * 70)
        print(f"Person: None (Error occurred)")
        print(f"Confidence: 0.00%")
        print(f"Status: ERROR")
        print("-" * 70)
        print(f"\nACCESS DENIED: Error during voice validation")
        print(f"Error: {str(e)}")
        print("\n" + "=" * 70)
        print("Product recommendation will NOT be displayed due to voice validation failure.")
        print("=" * 70)
        return False


def simulate_unauthorized_attempt(image_name_or_path=None, audio_name_or_path=None):
    """
    Simulate an unauthorized access attempt (unknown face or voice)
    
    Parameters:
    -----------
    image_name_or_path : str or Path, optional
        Image name or path to unauthorized image. If None, uses a default unauthorized image.
    audio_name_or_path : str or Path, optional
        Audio name or path to unauthorized audio. If None, uses a default unauthorized audio.
    """
    print_header("SYSTEM SIMULATION: UNAUTHORIZED ACCESS ATTEMPT")
    print("Demonstrating security: Attempting access with unauthorized credentials\n")
    
    # Use default unauthorized files if not provided
    root_dir = Path(__file__).parent.parent
    
    if image_name_or_path is None:
        # For unauthorized attempt, user must provide an image that's not in training set
        # We'll use a default image but warn the user
        images_dir = root_dir / "Image_Processing" / "Images"
        # Try to find any image file in the Images directory
        image_files = list(images_dir.glob("*.jpg")) if images_dir.exists() else []
        if image_files:
            # Use the first available image (user should provide a non-trained image)
            image_path = image_files[0]
            print("Note: Using first available image from Images directory.")
            print("For true unauthorized attempt, provide an image name that is NOT in the training set.")
        else:
            print("Error: No image files found in Images directory.")
            print("Please provide an image file using --image argument.")
            return
    else:
        image_path = find_image_file(image_name_or_path, root_dir)
        if not image_path:
            print(f"Error: Image file not found: {image_name_or_path}")
            return
    
    if audio_name_or_path is None:
        # Use any audio file (we'll test if it's unauthorized)
        audio_path = root_dir / "Audio_Processing" / "Audios" / "Erneste_yes_approve.wav"
    else:
        audio_path = find_audio_file(audio_name_or_path, root_dir)
        if not audio_path:
            print(f"Error: Audio file not found: {audio_name_or_path}")
            return
    
    # Step 1: Try Facial Recognition
    print_section("STEP 1: FACIAL RECOGNITION (UNAUTHORIZED)")
    print(f"Processing image: {image_path}")
    
    try:
        person, face_confidence = predict_face(image_path)
        
        if person is None:
            print("\nSECURITY WORKING: Unauthorized face detected and blocked")
            print(f"Result: Access denied (confidence: {face_confidence:.2%})")
            print("\n" + "=" * 70)
            print("SYSTEM SECURITY: Unauthorized access attempt blocked at facial recognition stage")
            print("=" * 70)
            return
        
        # If somehow recognized, check if it's a known user
        print(f"\nFace recognized as: {person} (confidence: {face_confidence:.2%})")
        print("Continuing to voice validation...")
        
    except Exception as e:
        print(f"\nSECURITY WORKING: Error during recognition (likely unauthorized)")
        print(f"Error: {str(e)}")
        print("\n" + "=" * 70)
        return
    
    # Step 2: Try Voice Validation
    print_section("STEP 2: VOICE VALIDATION (UNAUTHORIZED)")
    print(f"Processing audio: {audio_path}")
    
    try:
        voice_person, voice_confidence = predict_voice(audio_path)
        
        if voice_person is None:
            print("\nSECURITY WORKING: Unauthorized voice detected and blocked")
            print(f"Result: Access denied (confidence: {voice_confidence:.2%})")
            print("\n" + "=" * 70)
            print("SYSTEM SECURITY: Unauthorized access attempt blocked at voice validation stage")
            print("=" * 70)
            return
        
        # Check if voice matches face
        if voice_person != person:
            print(f"\nSECURITY WORKING: Voice does not match face identity")
            print(f"Face: {person}, Voice: {voice_person}")
            print("\n" + "=" * 70)
            print("SYSTEM SECURITY: Unauthorized access attempt blocked - identity mismatch")
            print("=" * 70)
            return
        
        print(f"\nVoice recognized as: {voice_person} (confidence: {voice_confidence:.2%})")
        print("Note: This is a known user. For true unauthorized attempt, use unknown voice/face.")
        
    except Exception as e:
        print(f"\nSECURITY WORKING: Error during voice validation")
        print(f"Error: {str(e)}")
        print("\n" + "=" * 70)
        return


def find_image_file(image_name_or_path, root_dir):
    """
    Find image file by name or path.
    Searches in both original Images folder and Augmented_Images folder.
    
    Parameters:
    -----------
    image_name_or_path : str
        Image filename (e.g., "Erneste_Neutral.jpg") or full path
    root_dir : Path
        Root directory of the project
    
    Returns:
    --------
    Path : Path to the image file, or None if not found
    """
    if not image_name_or_path:
        return None
    
    # First, try as a filename in the standard Images directory
    images_dir = root_dir / "Image_Processing" / "Images"
    image_path = images_dir / image_name_or_path
    
    if image_path.exists():
        return image_path
    
    # Try in Augmented_Images directory
    augmented_images_dir = root_dir / "Image_Processing" / "Augmented_Images"
    image_path = augmented_images_dir / image_name_or_path
    
    if image_path.exists():
        return image_path
    
    # If not found, try as a full path
    image_path = Path(image_name_or_path)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    
    # Try relative to project root
    image_path = root_dir / image_name_or_path
    if image_path.exists():
        return image_path
    
    return None


def find_audio_file(audio_name_or_path, root_dir):
    """
    Find audio file by name or path.
    Searches in both original Audios folder and Augmented_Audios folder.
    
    Parameters:
    -----------
    audio_name_or_path : str
        Audio filename (e.g., "Erneste_yes_approve.wav") or full path
    root_dir : Path
        Root directory of the project
    
    Returns:
    --------
    Path : Path to the audio file, or None if not found
    """
    if not audio_name_or_path:
        return None
    
    # First, try as a filename in the standard Audios directory
    audios_dir = root_dir / "Audio_Processing" / "Audios"
    audio_path = audios_dir / audio_name_or_path
    
    if audio_path.exists():
        return audio_path
    
    # Try in Augmented_Audios directory
    augmented_audios_dir = root_dir / "Audio_Processing" / "Augmented_Audios"
    audio_path = augmented_audios_dir / audio_name_or_path
    
    if audio_path.exists():
        return audio_path
    
    # If not found, try as a full path
    audio_path = Path(audio_name_or_path)
    if audio_path.is_absolute() and audio_path.exists():
        return audio_path
    
    # Try relative to project root
    audio_path = root_dir / audio_name_or_path
    if audio_path.exists():
        return audio_path
    
    return None


def get_user_inputs():
    """
    Interactive function to get user inputs step by step.
    
    Returns:
    --------
    tuple : (image_path, customer_data, audio_path)
    """
    root_dir = Path(__file__).parent.parent
    images_dir = root_dir / "Image_Processing" / "Images"
    audios_dir = root_dir / "Audio_Processing" / "Audios"
    
    print_header("INTERACTIVE SYSTEM SIMULATION")
    print("Please provide the following information:\n")
    
    # Step 1: Get image name
    print_section("STEP 1: Enter Image Name")
    # List available images from both directories
    augmented_images_dir = root_dir / "Image_Processing" / "Augmented_Images"
    available_images = []
    
    if images_dir.exists():
        available_images.extend([f.name for f in images_dir.glob("*.jpg")])
    if augmented_images_dir.exists():
        available_images.extend([f.name for f in augmented_images_dir.glob("*.jpg")])
    
    if available_images:
        available_images = sorted(set(available_images))  # Remove duplicates
        print(f"Available images (from Images and Augmented_Images folders):")
        for img in available_images[:20]:  # Show first 20 to avoid too long list
            print(f"  - {img}")
        if len(available_images) > 20:
            print(f"  ... and {len(available_images) - 20} more images")
        print()
    
    while True:
        image_input = input("Enter the image name (e.g., 'Erneste_Neutral.jpg') or full path (or press Enter for default): ").strip()
        
        if not image_input:
            # Use default
            image_path = images_dir / "Erneste_Neutral.jpg"
            if image_path.exists():
                print(f"Using default image: {image_path.name}\n")
                break
            else:
                print("Default image not found. Please enter an image name.\n")
                continue
        
        image_path = find_image_file(image_input, root_dir)
        
        if image_path and image_path.exists():
            folder_type = "Augmented_Images" if "Augmented_Images" in str(image_path) else "Images"
            print(f"Image found: {image_path.name} (from {folder_type})\n")
            break
        else:
            print(f"Error: Image file not found: {image_input}")
            print(f"Searched in: {images_dir} and {augmented_images_dir}")
            print("Please try again.\n")
    
    # CRITICAL: Run facial recognition NOW before asking for customer data
    print_section("FACIAL RECOGNITION CHECK")
    print("Verifying identity before proceeding...")
    print(f"Processing image: {image_path.name}")
    
    # Define authorized members
    AUTHORIZED_MEMBERS = ['Erneste', 'Thierry', 'Idara', 'Rodas']
    person = None
    face_confidence = 0.0
    
    try:
        # Import predict_face function
        from predict_face import predict_face
        
        # Run facial recognition
        person, face_confidence = predict_face(image_path)
        
        # Display the result
        print("\n" + "-" * 70)
        print("FACIAL RECOGNITION RESULT:")
        print("-" * 70)
        if person is None or person == "":
            print(f"Person: None (Not Recognized)")
            print(f"Confidence: {face_confidence:.2%}")
            print(f"Status: FAILED")
            print(f"Decision: STOP - Person not found in training data")
        else:
            print(f"Person: {person}")
            print(f"Confidence: {face_confidence:.2%}")
            if person in AUTHORIZED_MEMBERS:
                print(f"Status: SUCCESS")
                print(f"Decision: CONTINUE - Person is authorized")
            else:
                print(f"Status: FAILED")
                print(f"Decision: STOP - Person not in authorized list")
        print("-" * 70)
        
        # Check if we should continue
        if person is None or person == "":
            print("\n" + "=" * 70)
            print("ACCESS DENIED: Facial recognition failed")
            print("=" * 70)
            print(f"Reason: Person not recognized")
            print(f"Confidence: {face_confidence:.2%}")
            print(f"The person in this image was not found in the training data.")
            print(f"Authorized members: {', '.join(AUTHORIZED_MEMBERS)}")
            print("\n" + "=" * 70)
            print("SYSTEM SECURITY: Access denied at facial recognition stage")
            print("Product recommendation will NOT be executed.")
            print("=" * 70)
            return None, None, None, None, None  # Return None to indicate failure
        
        # Check if person is authorized
        if person not in AUTHORIZED_MEMBERS:
            print("\n" + "=" * 70)
            print("ACCESS DENIED: Person not authorized")
            print("=" * 70)
            print(f"Reason: '{person}' is not in the authorized members list")
            print(f"Confidence: {face_confidence:.2%}")
            print(f"Authorized members: {', '.join(AUTHORIZED_MEMBERS)}")
            print(f"The system only allows access to trained members.")
            print("\n" + "=" * 70)
            print("SYSTEM SECURITY: Access denied at facial recognition stage")
            print("Product recommendation will NOT be executed.")
            print("=" * 70)
            return None, None, None, None, None  # Return None to indicate failure
        
        # Success - person is recognized and authorized
        print(f"\nFACIAL RECOGNITION SUCCESSFUL")
        print(f"Identified as: {person}")
        print(f"Confidence: {face_confidence:.2%}")
        print(f"Authorization: VERIFIED - Person is in authorized list")
        print(f"Proceeding to customer data collection...\n")
        
        # Store verified person and confidence for later use
        verified_person = person
        verified_confidence = face_confidence
        
    except Exception as e:
        print("\n" + "-" * 70)
        print("FACIAL RECOGNITION RESULT:")
        print("-" * 70)
        print(f"Person: None (Error occurred)")
        print(f"Confidence: 0.00%")
        print(f"Status: ERROR")
        print("-" * 70)
        print("\n" + "=" * 70)
        print("ACCESS DENIED: Error during facial recognition")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("\n" + "=" * 70)
        print("SYSTEM SECURITY: Access denied at facial recognition stage")
        print("Product recommendation will NOT be executed.")
        print("=" * 70)
        return None, None, None, None, None  # Return None to indicate failure
    
    # Step 2: Get customer data for product recommendation
    # Only reach here if facial recognition was successful
    print_section("STEP 2: Enter Customer Data for Product Recommendation")
    print("Enter the following customer transaction details:\n")
    
    customer_data = {}
    
    # Purchase amount
    while True:
        try:
            amount_input = input("Purchase amount (or press Enter for default 300.0): ").strip()
            customer_data['purchase_amount'] = float(amount_input) if amount_input else 300.0
            if customer_data['purchase_amount'] > 0:
                break
            else:
                print("Purchase amount must be positive. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Customer rating
    while True:
        try:
            rating_input = input("Customer rating (1-5, or press Enter for default 4.5): ").strip()
            customer_data['customer_rating'] = float(rating_input) if rating_input else 4.5
            if 1.0 <= customer_data['customer_rating'] <= 5.0:
                break
            else:
                print("Rating must be between 1 and 5. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
    
    # Engagement score
    while True:
        try:
            engagement_input = input("Engagement score (0-100, or press Enter for default 75.0): ").strip()
            customer_data['engagement_score'] = float(engagement_input) if engagement_input else 75.0
            if 0.0 <= customer_data['engagement_score'] <= 100.0:
                break
            else:
                print("Engagement score must be between 0 and 100. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 100.")
    
    # Purchase interest score
    while True:
        try:
            interest_input = input("Purchase interest score (1-5, or press Enter for default 3.5): ").strip()
            customer_data['purchase_interest_score'] = float(interest_input) if interest_input else 3.5
            if 1.0 <= customer_data['purchase_interest_score'] <= 5.0:
                break
            else:
                print("Purchase interest score must be between 1 and 5. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
    
    # Social media platform
    platforms = ['Facebook', 'Twitter', 'Instagram', 'LinkedIn', 'TikTok']
    print(f"\nAvailable platforms: {', '.join(platforms)}")
    while True:
        platform_input = input("Social media platform (or press Enter for default 'Facebook'): ").strip()
        if not platform_input:
            customer_data['social_media_platform'] = 'Facebook'
            break
        elif platform_input in platforms:
            customer_data['social_media_platform'] = platform_input
            break
        else:
            print(f"Invalid platform. Please choose from: {', '.join(platforms)}")
    
    # Review sentiment
    sentiments = ['Positive', 'Neutral', 'Negative']
    print(f"\nAvailable sentiments: {', '.join(sentiments)}")
    while True:
        sentiment_input = input("Review sentiment (or press Enter for default 'Positive'): ").strip()
        if not sentiment_input:
            customer_data['review_sentiment'] = 'Positive'
            break
        elif sentiment_input in sentiments:
            customer_data['review_sentiment'] = sentiment_input
            break
        else:
            print(f"Invalid sentiment. Please choose from: {', '.join(sentiments)}")
    
    # Purchase date
    while True:
        date_input = input("Purchase date (YYYY-MM-DD, or press Enter for default '2024-01-15'): ").strip()
        if not date_input:
            customer_data['purchase_date'] = '2024-01-15'
            break
        else:
            # Basic date validation
            try:
                from datetime import datetime
                datetime.strptime(date_input, '%Y-%m-%d')
                customer_data['purchase_date'] = date_input
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD format.")
    
    print("\nCustomer data collected successfully!\n")
    
    # Step 3: Get audio name
    print_section("STEP 3: Enter Voice/Audio Name")
    # List available audio files from both directories
    augmented_audios_dir = root_dir / "Audio_Processing" / "Augmented_Audios"
    available_audios = []
    
    if audios_dir.exists():
        available_audios.extend([f.name for f in audios_dir.glob("*.wav")])
    if augmented_audios_dir.exists():
        available_audios.extend([f.name for f in augmented_audios_dir.glob("*.wav")])
    
    if available_audios:
        available_audios = sorted(set(available_audios))  # Remove duplicates
        print(f"Available audio files (from Audios and Augmented_Audios folders):")
        for audio in available_audios[:20]:  # Show first 20 to avoid too long list
            print(f"  - {audio}")
        if len(available_audios) > 20:
            print(f"  ... and {len(available_audios) - 20} more audio files")
        print()
    
    while True:
        audio_input = input("Enter the audio name (e.g., 'Erneste_yes_approve.wav') or full path (or press Enter for default): ").strip()
        
        if not audio_input:
            # Use default
            audio_path = audios_dir / "Erneste_yes_approve.wav"
            if audio_path.exists():
                print(f"Using default audio: {audio_path.name}\n")
                break
            else:
                print("Default audio not found. Please enter an audio name.\n")
                continue
        
        audio_path = find_audio_file(audio_input, root_dir)
        
        if audio_path and audio_path.exists():
            folder_type = "Augmented_Audios" if "Augmented_Audios" in str(audio_path) else "Audios"
            print(f"Audio file found: {audio_path.name} (from {folder_type})\n")
            break
        else:
            print(f"Error: Audio file not found: {audio_input}")
            print(f"Searched in: {audios_dir} and {augmented_audios_dir}")
            print("Please try again.\n")
    
    return image_path, customer_data, audio_path, verified_person, verified_confidence


def main():
    """Main function to run system simulations"""
    parser = argparse.ArgumentParser(
        description="System Simulation: User Identity and Product Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Image name (e.g., "Erneste_Neutral.jpg") or full path to facial image for recognition'
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        help='Audio name (e.g., "Erneste_yes_approve.wav") or full path to audio file for voice validation'
    )
    
    parser.add_argument(
        '--unauthorized',
        action='store_true',
        help='Simulate an unauthorized access attempt'
    )
    
    parser.add_argument(
        '--amount',
        type=float,
        help='Purchase amount for product recommendation'
    )
    
    parser.add_argument(
        '--rating',
        type=float,
        help='Customer rating for product recommendation'
    )
    
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Use command-line arguments instead of interactive prompts'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent
    
    # Determine which simulation to run
    if args.unauthorized:
        # Simulate unauthorized attempt
        simulate_unauthorized_attempt(args.image, args.audio)
    else:
        # Check if we should use interactive mode or command-line args
        if not args.non_interactive and not args.image and not args.audio:
            # Interactive mode - get inputs from user
            result = get_user_inputs()
            if result is None or (isinstance(result, tuple) and result[0] is None):
                # Facial recognition failed in interactive mode
                print("\nTransaction cancelled - facial recognition failed")
                print("System will not proceed to product recommendation.")
                return
            image_path, customer_data, audio_path, verified_person, verified_confidence = result
            
            # Run full transaction with collected inputs
            # Pass verified_person to skip redundant facial recognition
            success = simulate_full_transaction(
                image_path, 
                audio_path, 
                customer_data,
                verified_person=verified_person,
                face_confidence=verified_confidence
            )
            
            if success:
                print("\nTransaction completed successfully!")
            else:
                print("\nTransaction failed - access denied")
            return
        else:
            # Use command-line arguments
            if args.image:
                image_path = find_image_file(args.image, root_dir)
                if not image_path:
                    print(f"Error: Image file not found: {args.image}")
                    print(f"Searched in: {root_dir / 'Image_Processing' / 'Images'}")
                    print(f"and: {root_dir / 'Image_Processing' / 'Augmented_Images'}")
                    return
            else:
                # Default: Use Erneste's neutral image
                image_path = root_dir / "Image_Processing" / "Images" / "Erneste_Neutral.jpg"
            
            if args.audio:
                audio_path = find_audio_file(args.audio, root_dir)
                if not audio_path:
                    print(f"Error: Audio file not found: {args.audio}")
                    print(f"Searched in: {root_dir / 'Audio_Processing' / 'Audios'}")
                    print(f"and: {root_dir / 'Audio_Processing' / 'Augmented_Audios'}")
                    return
            else:
                # Default: Use Erneste's approval audio
                audio_path = root_dir / "Audio_Processing" / "Audios" / "Erneste_yes_approve.wav"
            
            # Prepare customer data
            customer_data = None
            if args.amount or args.rating:
                customer_data = {
                    'purchase_amount': args.amount if args.amount else 300.0,
                    'customer_rating': args.rating if args.rating else 4.5,
                    'engagement_score': 75.0,
                    'purchase_interest_score': 3.5,
                    'social_media_platform': 'Facebook',
                    'review_sentiment': 'Positive',
                    'purchase_date': '2024-01-15'
                }
        
        # Check if files exist
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return
        
        # Run full transaction
        success = simulate_full_transaction(image_path, audio_path, customer_data)
        
        if success:
            print("\nTransaction completed successfully!")
        else:
            print("\n failed - access denied")


if __name__ == "__main__":
    main()

