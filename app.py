from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

import os
import requests
import numpy as np
import tifffile

from io import BytesIO
from datetime import datetime, timedelta, timezone


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# COPERNICUS CREDENTIALS
# ============================================================

CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "process/v1"
)


# ============================================================
# GET COPERNICUS ACCESS TOKEN
# ============================================================

def get_copernicus_token():

    if not CLIENT_ID:
        raise Exception(
            "COPERNICUS_CLIENT_ID is missing from .env"
        )

    if not CLIENT_SECRET:
        raise Exception(
            "COPERNICUS_CLIENT_SECRET is missing from .env"
        )

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    response = requests.post(
        TOKEN_URL,
        data=data,
        timeout=30,
    )

    response.raise_for_status()

    token_data = response.json()

    if "access_token" not in token_data:
        raise Exception(
            "No access token received from Copernicus"
        )

    return token_data["access_token"]


# ============================================================
# NDVI HEALTH CLASSIFICATION
# ============================================================

def classify_ndvi(ndvi):

    if ndvi is None:
        return "No Data"

    if ndvi < 0.20:
        return "Very Low"

    elif ndvi < 0.40:
        return "Low"

    elif ndvi < 0.60:
        return "Moderate"

    elif ndvi < 0.80:
        return "Healthy"

    else:
        return "Very Healthy"


# ============================================================
# NDVI DESCRIPTION - ENGLISH
# ============================================================

def ndvi_description(ndvi):

    if ndvi is None:
        return (
            "No valid satellite pixels were available "
            "for this location."
        )

    if ndvi < 0.20:

        return (
            "Very low vegetation activity. "
            "The area may contain bare soil, water, "
            "or severely stressed vegetation."
        )

    elif ndvi < 0.40:

        return (
            "Low vegetation activity. "
            "The crop may be under stress or "
            "vegetation cover may be sparse."
        )

    elif ndvi < 0.60:

        return (
            "Moderate vegetation activity. "
            "The crop shows developing vegetation."
        )

    elif ndvi < 0.80:

        return (
            "Healthy vegetation activity. "
            "The crop appears to have good vegetation cover."
        )

    else:

        return (
            "Very healthy vegetation activity "
            "with a strong vegetation response."
        )


# ============================================================
# XAI - NDVI EXPLANATION
# ============================================================

def generate_xai_explanation(
    ndvi,
    health_status,
    valid_pixels,
    cloud_limit,
    language="en"
):

    if ndvi is None:
        return {
            "title": "No Data",
            "summary": (
                "Satellite data was not sufficient "
                "to explain crop health."
            ),
            "why": (
                "No valid vegetation pixels were found."
            ),
            "action": (
                "Try another date with less cloud coverage."
            )
        }


    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    if language == "en":

        if ndvi < 0.20:

            return {
                "title": "Why is vegetation very low?",
                "summary": (
                    f"The NDVI value is {ndvi:.4f}, "
                    "which indicates very low vegetation activity."
                ),
                "why": (
                    "Low NDVI means the satellite detected "
                    "a weak difference between near-infrared "
                    "and red reflectance. This can happen "
                    "with bare soil, water, sparse vegetation, "
                    "or severely stressed crops."
                ),
                "action": (
                    "Check the field for water stress, "
                    "poor plant growth, bare soil, pests, "
                    "or crop damage."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04). "
                    "B08 is near-infrared and B04 is red."
                )
            }

        elif ndvi < 0.40:

            return {
                "title": "Why is vegetation low?",
                "summary": (
                    f"The NDVI value is {ndvi:.4f}, "
                    "showing relatively low vegetation activity."
                ),
                "why": (
                    "The vegetation response is weaker than "
                    "normally expected for dense green vegetation. "
                    "Possible reasons include early crop growth, "
                    "water stress, sparse vegetation, or crop stress."
                ),
                "action": (
                    "Check crop growth, soil moisture, "
                    "irrigation and possible pest or disease symptoms."
                ),
                "technical_reason": (
                    "The NDVI is calculated from Sentinel-2 "
                    "B08 near-infrared and B04 red bands."
                )
            }

        elif ndvi < 0.60:

            return {
                "title": "Why is vegetation moderate?",
                "summary": (
                    f"The NDVI value is {ndvi:.4f}, "
                    "indicating moderate vegetation activity."
                ),
                "why": (
                    "The satellite detects a moderate vegetation "
                    "response. This can occur when crops are "
                    "developing or when vegetation is not yet dense."
                ),
                "action": (
                    "Continue monitoring the crop. "
                    "Maintain suitable irrigation and nutrition."
                ),
                "technical_reason": (
                    "NDVI uses the contrast between near-infrared "
                    "and red reflectance."
                )
            }

        elif ndvi < 0.80:

            return {
                "title": "Why is the crop healthy?",
                "summary": (
                    f"The NDVI value is {ndvi:.4f}, "
                    "indicating healthy vegetation."
                ),
                "why": (
                    "Healthy green vegetation usually reflects "
                    "more near-infrared light and absorbs more "
                    "red light. This produces a stronger NDVI value."
                ),
                "action": (
                    "Continue the current crop management "
                    "and regularly monitor the field."
                ),
                "technical_reason": (
                    "The result is derived from Sentinel-2 "
                    "B08 near-infrared and B04 red bands."
                )
            }

        else:

            return {
                "title": "Why is vegetation very healthy?",
                "summary": (
                    f"The NDVI value is {ndvi:.4f}, "
                    "showing a strong vegetation response."
                ),
                "why": (
                    "Dense and actively growing green vegetation "
                    "generally has high near-infrared reflectance "
                    "and strong absorption in the red band."
                ),
                "action": (
                    "The vegetation appears strong. "
                    "Continue monitoring for changes over time."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)."
                )
            }


    # --------------------------------------------------------
    # MALAYALAM
    # --------------------------------------------------------

    elif language == "ml":

        if ndvi < 0.20:

            return {
                "title": "ചെടികളുടെ ആരോഗ്യനില വളരെ കുറവാണ്",
                "summary": (
                    f"NDVI മൂല്യം {ndvi:.4f} ആണ്. "
                    "ഇത് വളരെ കുറവ് സസ്യ പ്രവർത്തനമാണ് കാണിക്കുന്നത്."
                ),
                "why": (
                    "NDVI കുറവായതിനാൽ പ്രദേശത്ത് "
                    "സസ്യവളർച്ച കുറവായിരിക്കാം. "
                    "വരണ്ട മണ്ണ്, വെള്ളം, കുറവ് സസ്യാവരണം "
                    "അല്ലെങ്കിൽ വിളയിലെ സമ്മർദ്ദം എന്നിവ കാരണമാകാം."
                ),
                "action": (
                    "വയലിലെ വെള്ളത്തിന്റെ ലഭ്യത, "
                    "ചെടികളുടെ വളർച്ച, കീടബാധ, "
                    "വിളനാശം എന്നിവ പരിശോധിക്കുക."
                ),
                "technical_reason": (
                    "Sentinel-2 ഉപഗ്രഹത്തിലെ B08, B04 ബാൻഡുകൾ "
                    "ഉപയോഗിച്ചാണ് NDVI കണക്കാക്കുന്നത്."
                )
            }

        elif ndvi < 0.40:

            return {
                "title": "സസ്യവളർച്ച കുറവാണ്",
                "summary": (
                    f"NDVI {ndvi:.4f} ആണ്. "
                    "സസ്യ പ്രവർത്തനം കുറവാണ്."
                ),
                "why": (
                    "ചെടികളുടെ പച്ചപ്പ് കുറവായിരിക്കാം. "
                    "വിളയുടെ പ്രാരംഭ വളർച്ച, "
                    "വെള്ളക്കുറവ് അല്ലെങ്കിൽ മറ്റ് സമ്മർദ്ദങ്ങൾ "
                    "ഇതിന് കാരണമാകാം."
                ),
                "action": (
                    "ജലസേചനം, മണ്ണിലെ ഈർപ്പം, "
                    "കീടബാധ എന്നിവ പരിശോധിക്കുക."
                ),
                "technical_reason": (
                    "NDVI Sentinel-2 ലെ ചുവപ്പ്, "
                    "Near-Infrared ബാൻഡുകളിൽ നിന്നാണ് ലഭിക്കുന്നത്."
                )
            }

        elif ndvi < 0.60:

            return {
                "title": "സസ്യവളർച്ച മിതമായ നിലയിലാണ്",
                "summary": (
                    f"NDVI {ndvi:.4f} ആണ്. "
                    "വിള മിതമായ വളർച്ച കാണിക്കുന്നു."
                ),
                "why": (
                    "വിള വളരുന്ന ഘട്ടത്തിലായിരിക്കാം. "
                    "സസ്യാവരണം ഇപ്പോഴും പൂർണ്ണമായി വികസിച്ചിട്ടില്ല."
                ),
                "action": (
                    "വിള നിരീക്ഷിക്കുന്നത് തുടരുക. "
                    "ആവശ്യമായ വെള്ളവും പോഷകങ്ങളും നൽകുക."
                ),
                "technical_reason": (
                    "B08, B04 ബാൻഡുകളുടെ പ്രതിഫലന വ്യത്യാസമാണ് "
                    "NDVI ഉപയോഗിക്കുന്നത്."
                )
            }

        elif ndvi < 0.80:

            return {
                "title": "വിളയുടെ ആരോഗ്യം നല്ലതാണ്",
                "summary": (
                    f"NDVI {ndvi:.4f} ആണ്. "
                    "വിള നല്ല സസ്യവളർച്ച കാണിക്കുന്നു."
                ),
                "why": (
                    "ആരോഗ്യമുള്ള പച്ച സസ്യങ്ങൾ "
                    "Near-Infrared പ്രകാശം കൂടുതലായി പ്രതിഫലിപ്പിക്കുകയും "
                    "ചുവപ്പ് പ്രകാശം കൂടുതൽ ആഗിരണം ചെയ്യുകയും ചെയ്യുന്നു."
                ),
                "action": (
                    "നിലവിലുള്ള കൃഷിരീതി തുടരുക. "
                    "വിളയെ സ്ഥിരമായി നിരീക്ഷിക്കുക."
                ),
                "technical_reason": (
                    "Sentinel-2 B08 Near-Infrared, "
                    "B04 Red ബാൻഡുകൾ ഉപയോഗിച്ചാണ് NDVI."
                )
            }

        else:

            return {
                "title": "വിളയുടെ ആരോഗ്യം വളരെ മികച്ചതാണ്",
                "summary": (
                    f"NDVI {ndvi:.4f} ആണ്. "
                    "വളരെ ശക്തമായ സസ്യ പ്രതികരണമാണ് കാണുന്നത്."
                ),
                "why": (
                    "നല്ല വളർച്ചയുള്ള പച്ച സസ്യങ്ങൾ "
                    "Near-Infrared പ്രകാശം ശക്തമായി പ്രതിഫലിപ്പിക്കുന്നു."
                ),
                "action": (
                    "വിളയുടെ ഇപ്പോഴത്തെ പരിചരണം തുടരുക. "
                    "കാലക്രമത്തിൽ മാറ്റങ്ങൾ നിരീക്ഷിക്കുക."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }


    # --------------------------------------------------------
    # HINDI
    # --------------------------------------------------------

    elif language == "hi":

        if ndvi < 0.20:

            return {
                "title": "फसल की वनस्पति बहुत कम है",
                "summary": (
                    f"NDVI मान {ndvi:.4f} है। "
                    "यह बहुत कम वनस्पति गतिविधि दिखाता है।"
                ),
                "why": (
                    "कम NDVI का मतलब है कि क्षेत्र में "
                    "हरी वनस्पति कम हो सकती है। "
                    "सूखी मिट्टी, पानी, कम वनस्पति या "
                    "फसल पर तनाव इसका कारण हो सकता है।"
                ),
                "action": (
                    "खेत में पानी की उपलब्धता, "
                    "फसल की वृद्धि, कीट और नुकसान की जांच करें।"
                ),
                "technical_reason": (
                    "NDVI Sentinel-2 के B08 और B04 बैंड से "
                    "गणना किया जाता है।"
                )
            }

        elif ndvi < 0.40:

            return {
                "title": "वनस्पति गतिविधि कम है",
                "summary": (
                    f"NDVI {ndvi:.4f} है और वनस्पति गतिविधि कम है।"
                ),
                "why": (
                    "फसल में कम हरियाली या शुरुआती वृद्धि "
                    "हो सकती है। पानी की कमी भी एक कारण हो सकती है।"
                ),
                "action": (
                    "सिंचाई, मिट्टी की नमी और फसल की स्थिति जांचें।"
                ),
                "technical_reason": (
                    "NDVI लाल और Near-Infrared प्रकाश के "
                    "प्रतिबिंब के अंतर से प्राप्त होता है।"
                )
            }

        elif ndvi < 0.60:

            return {
                "title": "फसल की स्थिति मध्यम है",
                "summary": (
                    f"NDVI {ndvi:.4f} है। "
                    "फसल मध्यम वनस्पति गतिविधि दिखाती है।"
                ),
                "why": (
                    "फसल अभी विकास के चरण में हो सकती है "
                    "या वनस्पति पूरी तरह घनी नहीं हुई है।"
                ),
                "action": (
                    "फसल की निगरानी जारी रखें और "
                    "आवश्यक पानी तथा पोषक तत्व दें।"
                ),
                "technical_reason": (
                    "NDVI Sentinel-2 के B08 और B04 बैंड से बनाया जाता है।"
                )
            }

        elif ndvi < 0.80:

            return {
                "title": "फसल स्वस्थ है",
                "summary": (
                    f"NDVI {ndvi:.4f} है। "
                    "फसल में अच्छी हरियाली दिखाई दे रही है।"
                ),
                "why": (
                    "स्वस्थ हरे पौधे Near-Infrared प्रकाश को "
                    "अधिक प्रतिबिंबित करते हैं और लाल प्रकाश को "
                    "अधिक अवशोषित करते हैं।"
                ),
                "action": (
                    "वर्तमान फसल प्रबंधन जारी रखें और "
                    "नियमित निगरानी करें।"
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }

        else:

            return {
                "title": "फसल बहुत स्वस्थ है",
                "summary": (
                    f"NDVI {ndvi:.4f} है। "
                    "बहुत मजबूत वनस्पति प्रतिक्रिया दिखाई दे रही है।"
                ),
                "why": (
                    "घनी और स्वस्थ हरी वनस्पति "
                    "Near-Infrared प्रकाश को अधिक प्रतिबिंबित करती है।"
                ),
                "action": (
                    "वर्तमान देखभाल जारी रखें और समय के साथ "
                    "फसल में बदलाव की निगरानी करें।"
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }


    # --------------------------------------------------------
    # KANNADA
    # --------------------------------------------------------

    elif language == "kn":

        if ndvi < 0.20:

            return {
                "title": "ಬೆಳೆಯ ಸಸ್ಯವರ್ಗ ತುಂಬಾ ಕಡಿಮೆಯಾಗಿದೆ",
                "summary": (
                    f"NDVI ಮೌಲ್ಯ {ndvi:.4f} ಆಗಿದೆ. "
                    "ಸಸ್ಯ ಚಟುವಟಿಕೆ ತುಂಬಾ ಕಡಿಮೆಯಾಗಿದೆ."
                ),
                "why": (
                    "ಕಡಿಮೆ NDVI ಎಂದರೆ ಹಸಿರು ಸಸ್ಯವರ್ಗ ಕಡಿಮೆಯಾಗಿರಬಹುದು. "
                    "ನೀರಿನ ಕೊರತೆ, ಒಣ ಮಣ್ಣು ಅಥವಾ ಬೆಳೆ ಒತ್ತಡ "
                    "ಕಾರಣವಾಗಿರಬಹುದು."
                ),
                "action": (
                    "ನೀರಿನ ಲಭ್ಯತೆ, ಮಣ್ಣಿನ ತೇವಾಂಶ, "
                    "ಕೀಟಗಳು ಮತ್ತು ಬೆಳೆ ಬೆಳವಣಿಗೆಯನ್ನು ಪರಿಶೀಲಿಸಿ."
                ),
                "technical_reason": (
                    "Sentinel-2 ನ B08 ಮತ್ತು B04 ಬ್ಯಾಂಡ್‌ಗಳಿಂದ NDVI "
                    "ಲೆಕ್ಕ ಹಾಕಲಾಗುತ್ತದೆ."
                )
            }

        elif ndvi < 0.40:

            return {
                "title": "ಸಸ್ಯ ಬೆಳವಣಿಗೆ ಕಡಿಮೆಯಾಗಿದೆ",
                "summary": (
                    f"NDVI {ndvi:.4f} ಆಗಿದೆ."
                ),
                "why": (
                    "ಬೆಳೆಯಲ್ಲಿ ಹಸಿರು ಸಸ್ಯವರ್ಗ ಕಡಿಮೆಯಾಗಿರಬಹುದು "
                    "ಅಥವಾ ಬೆಳೆ ಆರಂಭಿಕ ಬೆಳವಣಿಗೆಯಲ್ಲಿರಬಹುದು."
                ),
                "action": (
                    "ನೀರಾವರಿ, ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು "
                    "ಬೆಳೆಯ ಆರೋಗ್ಯವನ್ನು ಪರಿಶೀಲಿಸಿ."
                ),
                "technical_reason": (
                    "NDVI ಕೆಂಪು ಮತ್ತು Near-Infrared ಬೆಳಕಿನ "
                    "ಪ್ರತಿಫಲನದ ಆಧಾರದ ಮೇಲೆ ಲೆಕ್ಕಿಸಲಾಗುತ್ತದೆ."
                )
            }

        elif ndvi < 0.60:

            return {
                "title": "ಬೆಳೆಯ ಆರೋಗ್ಯ ಮಧ್ಯಮವಾಗಿದೆ",
                "summary": (
                    f"NDVI {ndvi:.4f} ಆಗಿದೆ."
                ),
                "why": (
                    "ಬೆಳೆ ಬೆಳವಣಿಗೆಯ ಹಂತದಲ್ಲಿರಬಹುದು "
                    "ಅಥವಾ ಸಸ್ಯವರ್ಗ ಇನ್ನೂ ದಟ್ಟವಾಗಿಲ್ಲ."
                ),
                "action": (
                    "ಬೆಳೆಯನ್ನು ನಿರಂತರವಾಗಿ ಗಮನಿಸಿ. "
                    "ಅಗತ್ಯ ನೀರು ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ನೀಡಿ."
                ),
                "technical_reason": (
                    "B08 ಮತ್ತು B04 ಬ್ಯಾಂಡ್‌ಗಳ ನಡುವಿನ ವ್ಯತ್ಯಾಸದಿಂದ NDVI ಸಿಗುತ್ತದೆ."
                )
            }

        elif ndvi < 0.80:

            return {
                "title": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ",
                "summary": (
                    f"NDVI {ndvi:.4f} ಆಗಿದೆ."
                ),
                "why": (
                    "ಆರೋಗ್ಯಕರ ಹಸಿರು ಸಸ್ಯಗಳು Near-Infrared ಬೆಳಕನ್ನು "
                    "ಹೆಚ್ಚಾಗಿ ಪ್ರತಿಫಲಿಸುತ್ತವೆ."
                ),
                "action": (
                    "ಪ್ರಸ್ತುತ ಬೆಳೆ ನಿರ್ವಹಣೆಯನ್ನು ಮುಂದುವರಿಸಿ."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }

        else:

            return {
                "title": "ಬೆಳೆ ತುಂಬಾ ಆರೋಗ್ಯಕರವಾಗಿದೆ",
                "summary": (
                    f"NDVI {ndvi:.4f} ಆಗಿದೆ."
                ),
                "why": (
                    "ದಟ್ಟವಾದ ಹಸಿರು ಸಸ್ಯವರ್ಗವು Near-Infrared ಬೆಳಕನ್ನು "
                    "ಹೆಚ್ಚಾಗಿ ಪ್ರತಿಫಲಿಸುತ್ತದೆ."
                ),
                "action": (
                    "ಪ್ರಸ್ತುತ ಆರೈಕೆಯನ್ನು ಮುಂದುವರಿಸಿ ಮತ್ತು "
                    "ಬೆಳೆಯ ಬದಲಾವಣೆಗಳನ್ನು ಗಮನಿಸಿ."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }


    # --------------------------------------------------------
    # TELUGU
    # --------------------------------------------------------

    elif language == "te":

        if ndvi < 0.20:

            return {
                "title": "పంటలో వృక్షసంపద చాలా తక్కువగా ఉంది",
                "summary": (
                    f"NDVI విలువ {ndvi:.4f}. "
                    "వృక్ష కార్యకలాపం చాలా తక్కువగా ఉంది."
                ),
                "why": (
                    "తక్కువ NDVI అంటే పచ్చని వృక్షసంపద తక్కువగా "
                    "ఉండవచ్చు. నీటి కొరత, ఎండిన నేల లేదా "
                    "పంట ఒత్తిడి కారణం కావచ్చు."
                ),
                "action": (
                    "నీటి లభ్యత, నేల తేమ, "
                    "పంట పెరుగుదల మరియు పురుగులను పరిశీలించండి."
                ),
                "technical_reason": (
                    "Sentinel-2 B08 మరియు B04 బ్యాండ్లతో NDVI లెక్కించబడుతుంది."
                )
            }

        elif ndvi < 0.40:

            return {
                "title": "వృక్ష కార్యకలాపం తక్కువగా ఉంది",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "పంటలో పచ్చదనం తక్కువగా ఉండవచ్చు "
                    "లేదా పంట ప్రారంభ దశలో ఉండవచ్చు."
                ),
                "action": (
                    "నీటిపారుదల, నేల తేమ మరియు "
                    "పంట ఆరోగ్యాన్ని పరిశీలించండి."
                ),
                "technical_reason": (
                    "NDVI ఎరుపు మరియు Near-Infrared "
                    "ప్రతిబింబాల ఆధారంగా లెక్కించబడుతుంది."
                )
            }

        elif ndvi < 0.60:

            return {
                "title": "పంట ఆరోగ్యం మధ్యస్థంగా ఉంది",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "పంట అభివృద్ధి దశలో ఉండవచ్చు "
                    "లేదా వృక్షసంపద ఇంకా దట్టంగా ఉండకపోవచ్చు."
                ),
                "action": (
                    "పంటను నిరంతరం పర్యవేక్షించండి. "
                    "అవసరమైన నీరు మరియు పోషకాలను అందించండి."
                ),
                "technical_reason": (
                    "B08 మరియు B04 బ్యాండ్ల మధ్య వ్యత్యాసంతో NDVI లెక్కించబడుతుంది."
                )
            }

        elif ndvi < 0.80:

            return {
                "title": "పంట ఆరోగ్యంగా ఉంది",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "ఆరోగ్యకరమైన పచ్చని మొక్కలు "
                    "Near-Infrared కాంతిని ఎక్కువగా ప్రతిబింబిస్తాయి."
                ),
                "action": (
                    "ప్రస్తుత పంట నిర్వహణను కొనసాగించండి."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }

        else:

            return {
                "title": "పంట చాలా ఆరోగ్యంగా ఉంది",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "దట్టమైన పచ్చని వృక్షసంపద "
                    "Near-Infrared కాంతిని బలంగా ప్రతిబింబిస్తుంది."
                ),
                "action": (
                    "ప్రస్తుత సంరక్షణను కొనసాగించి "
                    "పంట మార్పులను గమనించండి."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }


    # --------------------------------------------------------
    # TAMIL
    # --------------------------------------------------------

    elif language == "ta":

        if ndvi < 0.20:

            return {
                "title": "பயிரின் தாவர வளர்ச்சி மிகவும் குறைவு",
                "summary": (
                    f"NDVI மதிப்பு {ndvi:.4f}. "
                    "தாவர செயல்பாடு மிகவும் குறைவாக உள்ளது."
                ),
                "why": (
                    "குறைந்த NDVI பசுமையான தாவரங்கள் குறைவாக "
                    "இருப்பதைக் காட்டலாம். நீர் பற்றாக்குறை, "
                    "வறண்ட மண் அல்லது பயிர் அழுத்தம் காரணமாக இருக்கலாம்."
                ),
                "action": (
                    "நீர் கிடைப்பது, மண் ஈரப்பதம், "
                    "பயிர் வளர்ச்சி மற்றும் பூச்சிகளை சரிபார்க்கவும்."
                ),
                "technical_reason": (
                    "Sentinel-2 B08 மற்றும் B04 பட்டைகள் மூலம் "
                    "NDVI கணக்கிடப்படுகிறது."
                )
            }

        elif ndvi < 0.40:

            return {
                "title": "தாவர செயல்பாடு குறைவாக உள்ளது",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "பயிரில் பசுமை குறைவாக இருக்கலாம் "
                    "அல்லது பயிர் ஆரம்ப வளர்ச்சி நிலையில் இருக்கலாம்."
                ),
                "action": (
                    "நீர்ப்பாசனம், மண் ஈரப்பதம் மற்றும் "
                    "பயிரின் ஆரோக்கியத்தை சரிபார்க்கவும்."
                ),
                "technical_reason": (
                    "சிவப்பு மற்றும் Near-Infrared ஒளியின் "
                    "பிரதிபலிப்புகளின் அடிப்படையில் NDVI கணக்கிடப்படுகிறது."
                )
            }

        elif ndvi < 0.60:

            return {
                "title": "பயிரின் ஆரோக்கியம் மிதமான நிலையில் உள்ளது",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "பயிர் வளர்ச்சி நிலையில் இருக்கலாம் "
                    "அல்லது தாவரங்கள் இன்னும் அடர்த்தியாக வளரவில்லை."
                ),
                "action": (
                    "பயிரை தொடர்ந்து கண்காணிக்கவும். "
                    "தேவையான நீர் மற்றும் ஊட்டச்சத்துகளை வழங்கவும்."
                ),
                "technical_reason": (
                    "B08 மற்றும் B04 பட்டைகளின் வேறுபாட்டை "
                    "பயன்படுத்தி NDVI கணக்கிடப்படுகிறது."
                )
            }

        elif ndvi < 0.80:

            return {
                "title": "பயிர் ஆரோக்கியமாக உள்ளது",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "ஆரோக்கியமான பச்சை தாவரங்கள் "
                    "Near-Infrared ஒளியை அதிகமாக பிரதிபலிக்கின்றன."
                ),
                "action": (
                    "தற்போதைய பயிர் பராமரிப்பை தொடரவும்."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }

        else:

            return {
                "title": "பயிர் மிகவும் ஆரோக்கியமாக உள்ளது",
                "summary": (
                    f"NDVI {ndvi:.4f}."
                ),
                "why": (
                    "அடர்த்தியான பச்சை தாவரங்கள் "
                    "Near-Infrared ஒளியை வலுவாக பிரதிபலிக்கின்றன."
                ),
                "action": (
                    "தற்போதைய பராமரிப்பை தொடரவும் "
                    "மற்றும் பயிரில் ஏற்படும் மாற்றங்களை கண்காணிக்கவும்."
                ),
                "technical_reason": (
                    "NDVI = (B08 - B04) / (B08 + B04)"
                )
            }


    # --------------------------------------------------------
    # FALLBACK TO ENGLISH
    # --------------------------------------------------------

    else:

        return generate_xai_explanation(
            ndvi,
            health_status,
            valid_pixels,
            cloud_limit,
            "en"
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({

        "project": "AgriSpectra",

        "status": "Backend is running",

        "sentinel": "Sentinel-2 L2A",

        "ndvi": "Enabled",

        "explainable_ai": "Enabled",

        "supported_languages": [
            "en",
            "ml",
            "hi",
            "kn",
            "te",
            "ta"
        ]

    })


# ============================================================
# TEST COPERNICUS CONNECTION
# ============================================================

@app.route("/test-copernicus")
def test_copernicus():

    try:

        token = get_copernicus_token()

        return jsonify({

            "success": True,

            "message":
                "Successfully connected to "
                "Copernicus Data Space",

            "token_received":
                bool(token),

        })

    except requests.exceptions.RequestException as e:

        return jsonify({

            "success": False,

            "message":
                "Copernicus connection failed",

            "error":
                str(e),

        }), 500

    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                "Copernicus authentication failed",

            "error":
                str(e),

        }), 500


# ============================================================
# CREATE EVALSCRIPT
# ============================================================

def create_evalscript():

    return """

    //VERSION=3

    function setup() {

        return {

            input: [{
                bands: [
                    "B04",
                    "B08",
                    "dataMask"
                ]
            }],

            output: {

                bands: 2,

                sampleType: "FLOAT32"

            }
        };
    }


    function evaluatePixel(sample) {

        let denominator =
            sample.B08 + sample.B04;


        if (
            sample.dataMask === 0 ||
            denominator === 0
        ) {

            return [
                -9999,
                0
            ];
        }


        let ndvi =
            (
                sample.B08 -
                sample.B04
            ) / denominator;


        return [
            ndvi,
            sample.dataMask
        ];
    }

    """


# ============================================================
# REQUEST SENTINEL-2 DATA
# ============================================================

def request_sentinel_image(
    token,
    bbox,
    from_date,
    to_date,
    cloud_limit
):

    evalscript = create_evalscript()

    request_data = {

        "input": {

            "bounds": {

                "properties": {

                    "crs":
                        "http://www.opengis.net/"
                        "def/crs/OGC/1.3/CRS84"

                },

                "bbox": bbox,

            },

            "data": [

                {

                    "type":
                        "sentinel-2-l2a",

                    "dataFilter": {

                        "timeRange": {

                            "from":
                                from_date,

                            "to":
                                to_date,

                        },

                        "maxCloudCoverage":
                            cloud_limit,

                        "mosaickingOrder":
                            "leastCC",

                    }

                }

            ]

        },

        "output": {

            "width": 100,

            "height": 100,

            "responses": [

                {

                    "identifier":
                        "default",

                    "format": {

                        "type":
                            "image/tiff"

                    }

                }

            ]

        },

        "evalscript":
            evalscript,

    }


    response = requests.post(

        PROCESS_URL,

        headers={

            "Authorization":
                f"Bearer {token}",

            "Content-Type":
                "application/json",

            "Accept":
                "image/tiff",

        },

        json=request_data,

        timeout=180,

    )


    if not response.ok:

        raise Exception(
            f"Sentinel-2 request failed "
            f"({response.status_code}): "
            f"{response.text[:1000]}"
        )


    return response.content


# ============================================================
# DECODE TIFF
# ============================================================

def decode_ndvi_tiff(content):

    image = tifffile.imread(
        BytesIO(content)
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )


    print(
        "Sentinel TIFF shape:",
        image.shape
    )


    # --------------------------------------------------------
    # (2, height, width)
    # --------------------------------------------------------

    if (
        image.ndim == 3
        and image.shape[0] == 2
    ):

        ndvi_array = image[0]

        mask_array = image[1]


    # --------------------------------------------------------
    # (height, width, 2)
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[-1] == 2
    ):

        ndvi_array = image[:, :, 0]

        mask_array = image[:, :, 1]


    else:

        raise Exception(
            f"Unexpected TIFF structure: "
            f"{image.shape}"
        )


    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    valid_pixels = (

        (mask_array > 0)

        &

        (ndvi_array >= -1.0)

        &

        (ndvi_array <= 1.0)

        &

        np.isfinite(ndvi_array)

    )


    valid_ndvi = (
        ndvi_array[valid_pixels]
    )


    return valid_ndvi


# ============================================================
# EXPLAIN NDVI API
#
# This can be called separately from Flutter:
#
# /explain-ndvi?ndvi=0.72&language=ml
#
# ============================================================

@app.route(
    "/explain-ndvi",
    methods=["GET", "POST"]
)
def explain_ndvi():

    try:

        # ----------------------------------------------------
        # GET DATA
        # ----------------------------------------------------

        if request.method == "POST":

            data = (
                request.get_json(
                    silent=True
                ) or {}
            )

            ndvi_value = data.get("ndvi")

            language = data.get(
                "language",
                "en"
            )

            valid_pixels = int(
                data.get(
                    "valid_pixels",
                    0
                )
            )

            cloud_limit = int(
                data.get(
                    "cloud_limit",
                    30
                )
            )

        else:

            ndvi_value = request.args.get(
                "ndvi"
            )

            language = request.args.get(
                "language",
                "en"
            )

            valid_pixels = int(
                request.args.get(
                    "valid_pixels",
                    "0"
                )
            )

            cloud_limit = int(
                request.args.get(
                    "cloud_limit",
                    "30"
                )
            )


        # ----------------------------------------------------
        # VALIDATE NDVI
        # ----------------------------------------------------

        if ndvi_value is None:

            return jsonify({

                "success": False,

                "error":
                    "NDVI value is required."

            }), 400


        ndvi_value = float(
            ndvi_value
        )


        if (
            ndvi_value < -1
            or ndvi_value > 1
        ):

            return jsonify({

                "success": False,

                "error":
                    "NDVI must be between -1 and 1."

            }), 400


        # ----------------------------------------------------
        # HEALTH
        # ----------------------------------------------------

        health_status = classify_ndvi(
            ndvi_value
        )


        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------

        explanation = generate_xai_explanation(

            ndvi_value,

            health_status,

            valid_pixels,

            cloud_limit,

            language

        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "ndvi": round(
                ndvi_value,
                4
            ),

            "health_status":
                health_status,

            "language":
                language,

            "explainable_ai":
                explanation,

            "formula":
                "(B08 - B04) / (B08 + B04)",

            "meaning":
                "NDVI measures vegetation greenness "
                "using red and near-infrared reflectance."

        })


    except ValueError as e:

        return jsonify({

            "success": False,

            "error":
                "NDVI must be a number.",

            "details":
                str(e)

        }), 400


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# SENTINEL-2 NDVI
# ============================================================

@app.route(
    "/sentinel-ndvi",
    methods=["GET", "POST"]
)
def sentinel_ndvi():

    try:

        # ====================================================
        # GET LOCATION
        # ====================================================

        if request.method == "POST":

            data = (
                request.get_json(
                    silent=True
                ) or {}
            )

            if (
                "latitude" not in data
                or "longitude" not in data
            ):

                return jsonify({

                    "success": False,

                    "error":
                        "Latitude and longitude "
                        "are required.",

                }), 400


            latitude = float(
                data["latitude"]
            )

            longitude = float(
                data["longitude"]
            )

        else:

            latitude = float(
                request.args.get(
                    "latitude",
                    "11.2588"
                )
            )

            longitude = float(
                request.args.get(
                    "longitude",
                    "75.7804"
                )
            )


        # ====================================================
        # LANGUAGE
        # ====================================================

        if request.method == "POST":

            language = data.get(
                "language",
                "en"
            )

        else:

            language = request.args.get(
                "language",
                "en"
            )


        # ====================================================
        # VALIDATE COORDINATES
        # ====================================================

        if (
            latitude < -90
            or latitude > 90
        ):

            return jsonify({

                "success": False,

                "error":
                    "Invalid latitude",

            }), 400


        if (
            longitude < -180
            or longitude > 180
        ):

            return jsonify({

                "success": False,

                "error":
                    "Invalid longitude",

            }), 400


        # ====================================================
        # FARM BOUNDING BOX
        # ====================================================

        # Approximately 1 km around farm

        delta = 0.005


        bbox = [

            longitude - delta,

            latitude - delta,

            longitude + delta,

            latitude + delta,

        ]


        print(
            "\n===================================="
        )

        print(
            "AgriSpectra Sentinel-2 Analysis"
        )

        print(
            "Latitude:",
            latitude
        )

        print(
            "Longitude:",
            longitude
        )

        print(
            "Language:",
            language
        )

        print(
            "BBOX:",
            bbox
        )

        print(
            "===================================="
        )


        # ====================================================
        # GET TOKEN
        # ====================================================

        token = get_copernicus_token()


        # ====================================================
        # SEARCH STRATEGIES
        # ====================================================

        today = datetime.now(
            timezone.utc
        )


        search_strategies = [

            {
                "days": 60,
                "cloud": 30,
            },

            {
                "days": 120,
                "cloud": 60,
            },

            {
                "days": 180,
                "cloud": 80,
            },

            {
                "days": 365,
                "cloud": 90,
            },

        ]


        successful_ndvi = None

        selected_from_date = None

        selected_to_date = None

        selected_cloud_limit = None

        attempts = []


        # ====================================================
        # TRY SENTINEL-2 SEARCHES
        # ====================================================

        for strategy in search_strategies:

            days = strategy["days"]

            cloud_limit = strategy["cloud"]


            start_date = (
                today -
                timedelta(days=days)
            )


            from_date = (
                start_date.strftime(
                    "%Y-%m-%dT00:00:00Z"
                )
            )


            to_date = (
                today.strftime(
                    "%Y-%m-%dT23:59:59Z"
                )
            )


            print(
                f"\nTrying Sentinel-2:"
            )

            print(
                f"Days: {days}"
            )

            print(
                f"Cloud limit: {cloud_limit}%"
            )


            try:

                content = request_sentinel_image(

                    token,

                    bbox,

                    from_date,

                    to_date,

                    cloud_limit,

                )


                print(
                    "Image received:",
                    len(content),
                    "bytes"
                )


                valid_ndvi = (
                    decode_ndvi_tiff(
                        content
                    )
                )


                print(
                    "Valid pixels:",
                    valid_ndvi.size
                )


                attempts.append({

                    "days":
                        days,

                    "cloud_limit":
                        cloud_limit,

                    "valid_pixels":
                        int(
                            valid_ndvi.size
                        ),

                })


                if valid_ndvi.size > 0:

                    successful_ndvi = (
                        valid_ndvi
                    )

                    selected_from_date = (
                        from_date
                    )

                    selected_to_date = (
                        to_date
                    )

                    selected_cloud_limit = (
                        cloud_limit
                    )


                    print(
                        "\nSUCCESS!"
                    )


                    break


            except Exception as e:

                print(
                    "Attempt failed:",
                    str(e)
                )


                attempts.append({

                    "days":
                        days,

                    "cloud_limit":
                        cloud_limit,

                    "error":
                        str(e),

                })


        # ====================================================
        # NO VALID DATA
        # ====================================================

        if successful_ndvi is None:

            return jsonify({

                "success": False,

                "message":
                    "No valid Sentinel-2 pixels "
                    "were available for this farm "
                    "after trying multiple date "
                    "and cloud-coverage ranges.",

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "bbox":
                    bbox,

                "attempts":
                    attempts,

            }), 404


        # ====================================================
        # NDVI STATISTICS
        # ====================================================

        mean_ndvi = float(
            np.mean(
                successful_ndvi
            )
        )


        min_ndvi = float(
            np.min(
                successful_ndvi
            )
        )


        max_ndvi = float(
            np.max(
                successful_ndvi
            )
        )


        median_ndvi = float(
            np.median(
                successful_ndvi
            )
        )


        std_ndvi = float(
            np.std(
                successful_ndvi
            )
        )


        # ====================================================
        # HEALTH
        # ====================================================

        health_status = (
            classify_ndvi(
                mean_ndvi
            )
        )


        description = (
            ndvi_description(
                mean_ndvi
            )
        )


        # ====================================================
        # XAI EXPLANATION
        # ====================================================

        xai_explanation = (
            generate_xai_explanation(

                mean_ndvi,

                health_status,

                int(
                    successful_ndvi.size
                ),

                selected_cloud_limit,

                language

            )
        )


        # ====================================================
        # HEALTH PERCENTAGE
        # ====================================================

        health_percentage = (
            ((mean_ndvi + 1) / 2) * 100
        )

        health_percentage = max(
            0,
            min(
                100,
                health_percentage
            )
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        return jsonify({

            "success":
                True,

            "message":
                "Real Sentinel-2 NDVI data "
                "processed successfully.",

            "satellite":
                "Sentinel-2 L2A",

            "latitude":
                latitude,

            "longitude":
                longitude,

            "bbox":
                bbox,

            "date_from":
                selected_from_date,

            "date_to":
                selected_to_date,

            "cloud_coverage_limit":
                selected_cloud_limit,

            "ndvi":
                round(
                    mean_ndvi,
                    4
                ),

            "ndvi_min":
                round(
                    min_ndvi,
                    4
                ),

            "ndvi_max":
                round(
                    max_ndvi,
                    4
                ),

            "ndvi_median":
                round(
                    median_ndvi,
                    4
                ),

            "ndvi_std":
                round(
                    std_ndvi,
                    4
                ),

            "valid_pixels":
                int(
                    successful_ndvi.size
                ),

            "health_percentage":
                round(
                    health_percentage
                ),

            "health_status":
                health_status,

            "description":
                description,

            "language":
                language,

            # =================================================
            # EXPLAINABLE AI
            # =================================================

            "explainable_ai": {

                "enabled":
                    True,

                "title":
                    xai_explanation["title"],

                "summary":
                    xai_explanation["summary"],

                "why":
                    xai_explanation["why"],

                "action":
                    xai_explanation["action"],

                "technical_reason":
                    xai_explanation[
                        "technical_reason"
                    ],

            },

            # =================================================
            # FORMULA
            # =================================================

            "ndvi_formula":
                "(B08 - B04) / (B08 + B04)",

            "bands": {

                "B04":
                    "Red",

                "B08":
                    "Near-Infrared"

            },

            "search_attempts":
                attempts,

        })


    # ========================================================
    # ERRORS
    # ========================================================

    except ValueError as e:

        return jsonify({

            "success": False,

            "error":
                "Latitude, longitude or other numeric "
                "values must be valid numbers.",

            "details":
                str(e),

        }), 400


    except requests.exceptions.Timeout:

        return jsonify({

            "success": False,

            "error":
                "Sentinel-2 request timed out.",

        }), 504


    except requests.exceptions.RequestException as e:

        return jsonify({

            "success": False,

            "error":
                str(e),

        }), 500


    except Exception as e:

        print(
            "SERVER ERROR:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                str(e),

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )