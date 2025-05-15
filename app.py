import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback
import io
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

# For patching if needed
try:
    import patchify
except ImportError:
    patchify = None

# Munsell Color Reference Data
MUNSELL_COLORS = {
  "5R 8/1": {
    "name": "White",
    "hex": "#EAE1DE",
    "description": "5R 8/1 - White with a reddish hue. Very high value and very low chroma. Often indicates parent material color or deposits like carbonates with a slight red staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Influence", "Possible Carbonate Presence", "Slight Iron Staining"]
  },
  "5R 8/2": {
    "name": "Pinkish White",
    "hex": "#E6D5D2",
    "description": "5R 8/2 - Pinkish White. Very high value and low chroma. Suggests a predominantly light-colored mineral base with minor influence of reddish iron compounds.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Minor Iron Oxides", "Good Aeration if color is inherent"]
  },
  "5R 8/3": {
    "name": "Pink",
    "hex": "#E2C9C5",
    "description": "5R 8/3 - Pink. High value, low to moderate chroma. Light reddish color, often associated with fine-grained sedimentary rocks or soils with finely disseminated hematite.",
    "properties": ["Light Colored Minerals", "Disseminated Hematite", "Low Organic Matter", "Well-Drained if color from oxidation"]
  },
  "5R 8/4": {
    "name": "Pink",
    "hex": "#DEBEBA",
    "description": "5R 8/4 - Pink. High value, moderate chroma. More pronounced pink color, indicating a greater presence or concentration of hematite in a light-colored matrix.",
    "properties": ["Moderate Hematite", "Well-Aerated", "Good Drainage", "Low Organic Matter"]
  },
  "5R 7/1": {
    "name": "Light Gray",
    "hex": "#D0C8C7",
    "description": "5R 7/1 - Light Gray with a reddish hue. High value, very low chroma. Predominantly gray with a hint of red, may indicate some iron staining on light-colored materials or a mix of minerals.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Iron Influence", "Parent Material Color"]
  },
  "5R 7/2": {
    "name": "Pinkish Gray",
    "hex": "#C9B9B7",
    "description": "5R 7/2 - Pinkish Gray. High value, low chroma. A muted reddish-gray, indicating a mix of light-colored minerals with some hematite.",
    "properties": ["Mixed Mineralogy", "Some Hematite", "Low Organic Matter", "Generally Well-Drained"]
  },
  "5R 7/3": {
    "name": "Pink",
    "hex": "#C3A circonstancesCA9",
    "description": "5R 7/3 - Pink. High value, low-moderate chroma. A pale but distinct pink, suggesting a fair amount of finely distributed hematite.",
    "properties": ["Finely Distributed Hematite", "Good Aeration", "Low Organic Matter"]
  },
  "5R 7/4": {
    "name": "Pink",
    "hex": "#BEA09C",
    "description": "5R 7/4 - Pink. High value, moderate chroma. Clear pink color, indicating visible hematite content within a lighter soil matrix.",
    "properties": ["Visible Hematite", "Well-Drained", "Oxidizing Environment"]
  },
  "5R 7/6": {
    "name": "Light Red",
    "hex": "#B78C86",
    "description": "5R 7/6 - Light Red. High value, moderate to high chroma. A brighter, light red color, suggesting a significant amount of hematite.",
    "properties": ["Significant Hematite", "Good Aeration and Drainage", "May be from Red Parent Materials"]
  },
  "5R 7/8": {
    "name": "Light Red",
    "hex": "#AF7870",
    "description": "5R 7/8 - Light Red. High value, high chroma. Strong light red color, indicating abundant, relatively pure hematite.",
    "properties": ["Abundant Hematite", "Well-Oxidized", "Often in Weathered Soils or Red Sediments"]
  },
  "5R 6/1": {
    "name": "Gray",
    "hex": "#B1ADA9",
    "description": "5R 6/1 - Gray with a reddish hue. Moderate value, very low chroma. A neutral gray with a slight red tint, can be due to parent material or mixed lighting conditions for observation.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Slight Reddish Tinge"]
  },
  "5R 6/2": {
    "name": "Pinkish Gray",
    "hex": "#A99D9A",
    "description": "5R 6/2 - Pinkish Gray. Moderate value, low chroma. A muted reddish color mixed with gray, indicating some iron oxidation but not dominant.",
    "properties": ["Some Oxidized Iron", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "5R 6/3": {
    "name": "Light Reddish Brown",
    "hex": "#A28D88",
    "description": "5R 6/3 - Light Reddish Brown. Moderate value, low-moderate chroma. Often seen in soils where red iron oxides are present but mixed with other minerals or some organic staining.",
    "properties": ["Mixed Iron Oxides and Minerals", "Fair Drainage", "Moderate Organic Matter possible"]
  },
  "5R 6/4": {
    "name": "Light Reddish Brown",
    "hex": "#9B7E79",
    "description": "5R 6/4 - Light Reddish Brown. Moderate value, moderate chroma. A more distinct reddish brown, indicating a good presence of hematite.",
    "properties": ["Clear Hematite Presence", "Well-Aerated", "Good Drainage"]
  },
  "5R 6/6": {
    "name": "Red",
    "hex": "#926A63",
    "description": "5R 6/6 - Red. Moderate value, moderate to high chroma. A clear red color, typical of soils with significant hematite accumulation.",
    "properties": ["Significant Hematite", "Well-Drained", "Often in B horizons of Alfisols or Ultisols"]
  },
  "5R 6/8": {
    "name": "Red",
    "hex": "#8A554E",
    "description": "5R 6/8 - Red. Moderate value, high chroma. A strong red color, indicating high concentration of hematite, common in tropical or highly weathered soils.",
    "properties": ["High Hematite Concentration", "Intense Weathering", "Well-Oxidized Environment"]
  },
  "5R 5/1": {
    "name": "Dark Gray",
    "hex": "#807D7A",
    "description": "5R 5/1 - Dark Gray with a reddish hue. Moderate to low value, very low chroma. Dark color could be from some organic matter or parent rock, with a subtle red hint.",
    "properties": ["Some Organic Matter or Dark Minerals", "Slight Reddish Influence", "Variable Drainage"]
  },
  "5R 5/2": {
    "name": "Weak Red",
    "hex": "#7C706D",
    "description": "5R 5/2 - Weak Red. Moderate to low value, low chroma. A dull, weak red, suggesting limited hematite or masking by other components.",
    "properties": ["Limited Hematite", "May have Organic Mixing", "Moderate Aeration"]
  },
  "5R 5/3": {
    "name": "Reddish Brown",
    "hex": "#7A6560",
    "description": "5R 5/3 - Reddish Brown. Moderate to low value, low-moderate chroma. Common color for soils with moderate hematite content and some organic matter.",
    "properties": ["Moderate Hematite", "Some Organic Matter", "Generally Good Structure"]
  },
  "5R 5/4": {
    "name": "Reddish Brown",
    "hex": "#785C56",
    "description": "5R 5/4 - Reddish Brown. Moderate to low value, moderate chroma. A distinct reddish-brown, indicative of well-aerated conditions and iron oxidation.",
    "properties": ["Well-Aerated", "Iron Oxidation (Hematite)", "Good Agricultural Potential often"]
  },
  "5R 5/6": {
    "name": "Red",
    "hex": "#754E47",
    "description": "5R 5/6 - Red. Moderate to low value, moderate to high chroma. A clear, strong red, suggesting high amounts of hematite.",
    "properties": ["High Hematite Content", "Good Drainage", "Often Associated with Clay Accumulation"]
  },
  "5R 5/8": {
    "name": "Red",
    "hex": "#723F37",
    "description": "5R 5/8 - Red. Moderate to low value, high chroma. A very strong, relatively dark red. Indicates very high concentration of hematite.",
    "properties": ["Very High Hematite", "Intensely Oxidized", "Common in Tropical Latosols/Oxisols"]
  },
  "5R 4/1": {
    "name": "Dark Reddish Gray",
    "hex": "#605D5B",
    "description": "5R 4/1 - Dark Reddish Gray. Low value, very low chroma. Dark color often due to organic matter, with a slight reddish hue from iron oxides.",
    "properties": ["Organic Matter Influence", "Subtle Iron Oxide Presence", "Moisture Retentive"]
  },
  "5R 4/2": {
    "name": "Dark Reddish Gray",
    "hex": "#5E5350",
    "description": "5R 4/2 - Dark Reddish Gray. Low value, low chroma. A dark soil with a muted reddish color, suggesting organic matter mixed with some hematite.",
    "properties": ["Organic Matter Rich", "Some Hematite", "Typically Fertile"]
  },
  "5R 4/3": {
    "name": "Dark Reddish Brown",
    "hex": "#5E4C47",
    "description": "5R 4/3 - Dark Reddish Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides contribute to the color.",
    "properties": ["Good Organic Content", "Moderate Hematite", "Good Soil Structure"]
  },
  "5R 4/4": {
    "name": "Dark Reddish Brown",
    "hex": "#5E453F",
    "description": "5R 4/4 - Dark Reddish Brown. Low value, moderate chroma. A rich, dark reddish-brown indicating good aeration, iron oxides, and organic matter.",
    "properties": ["Well-Aerated", "Productive Soil", "Balanced Organic and Iron Content"]
  },
  "5R 4/6": {
    "name": "Dark Red",
    "hex": "#5E3A33",
    "description": "5R 4/6 - Dark Red. Low value, moderate to high chroma. A strong dark red, suggesting high hematite content, possibly with some organic matter.",
    "properties": ["High Hematite", "May have Some Organic Matter", "Good Drainage"]
  },
  "5R 4/8": {
    "name": "Dark Red",
    "hex": "#5E2F26",
    "description": "5R 4/8 - Dark Red. Low value, high chroma. An intense dark red, indicative of very high hematite concentration.",
    "properties": ["Very High Hematite Concentration", "Strongly Oxidized", "Often in Weathered Parent Materials"]
  },
  "5R 3/1": {
    "name": "Dark Gray",
    "hex": "#4A4847",
    "description": "5R 3/1 - Dark Gray (almost black) with a reddish hue. Very low value, very low chroma. Very dark, organic-rich soil with a subtle hint of underlying reddish minerals.",
    "properties": ["High Organic Matter", "Slight Reddish Mineral Influence", "Often Poorly Drained if very dark and low chroma"]
  },
  "5R 3/2": {
    "name": "Dusky Red",
    "hex": "#49403D",
    "description": "5R 3/2 - Dusky Red. Very low value, low chroma. A very dark, muted red. High organic matter content strongly influences this color, mixed with hematite.",
    "properties": ["Very High Organic Matter", "Hematite Present", "Moisture Retentive", "Fertile"]
  },
  "5R 3/3": {
    "name": "Dusky Red",
    "hex": "#4B3C38",
    "description": "5R 3/3 - Dusky Red. Very low value, low-moderate chroma. Dark reddish color, suggesting a balance of organic matter and significant iron oxide (hematite).",
    "properties": ["Significant Organic Matter and Hematite", "Good Fertility", "Moist Conditions Possible"]
  },
  "5R 3/4": {
    "name": "Dusky Red",
    "hex": "#4D3630",
    "description": "5R 3/4 - Dusky Red. Very low value, moderate chroma. A darker red, less masked by organic matter than /2 or /3, but still dark. Strong hematite presence.",
    "properties": ["Strong Hematite Presence", "Moderate Organic Matter", "Well-Aerated for a dark soil"]
  },
  "5R 3/6": {
    "name": "Dark Red",
    "hex": "#4F2B23",
    "description": "5R 3/6 - Dark Red. Very low value, moderate to high chroma. A deep, dark red color, indicating very high hematite content with less organic matter influence than lower chromas.",
    "properties": ["Very High Hematite", "Good Drainage if structure allows", "Parent Material can be iron-rich rock"]
  },
  "5R 3/8": {
    "name": "Dark Red",
    "hex": "#502115",
    "description": "5R 3/8 - Dark Red. Very low value, high chroma. An intense, very dark red; nearly pure hematite coloration at a low value.",
    "properties": ["Concentrated Hematite", "Strongly Oxidized Environment", "Can be Hard and Dense"]
  },
  "5R 2.5/1": {
    "name": "Black",
    "hex": "#3B3A39",
    "description": "5R 2.5/1 - Black with a reddish hue. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or charcoal with a very faint hint of red.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal Possible", "Often Waterlogged", "Very Poorly Drained"]
  },
  "5R 2.5/2": {
    "name": "Reddish Black",
    "hex": "#3A3230",
    "description": "5R 2.5/2 - Reddish Black. Extremely low value, low chroma. Very dark soil where organic matter dominates, but with a noticeable reddish influence from iron oxides.",
    "properties": ["Dominant Organic Matter", "Noticeable Hematite", "Poor Drainage Common", "High Fertility when drained"]
  },
  "5R 2.5/3": {
    "name": "Reddish Black",
    "hex": "#3D2E2B",
    "description": "5R 2.5/3 - Reddish Black. Extremely low value, low-moderate chroma. A very dark soil with a more apparent reddish character than 2.5/2.",
    "properties": ["High Organic Matter", "Clearer Reddish Tinge from Hematite", "Moist Environment"]
  },
  "5R 2.5/4": {
    "name": "Dark Dusky Red",
    "hex": "#3F2924",
    "description": "5R 2.5/4 - Dark Dusky Red. Extremely low value, moderate chroma. Darkest expression of red before becoming black, significant hematite under very low value conditions.",
    "properties": ["Significant Hematite in Dark Matrix", "Some Organic Matter", "Low Light Reflectance"]
  },
  "5R 2.5/6": {
    "name": "Very Dusky Red",
    "hex": "#422018",
    "description": "5R 2.5/6 - Very Dusky Red. Extremely low value, moderate-high chroma. A very dark but relatively strong red, indicating a high concentration of hematite in a very dark soil.",
    "properties": ["Concentrated Hematite in Dark Matrix", "Uncommon in typical soils, may indicate specific mineral stains or coatings"]
  },
  "7.5R 8/1": {
    "name": "White",
    "hex": "#EAE2DF",
    "description": "7.5R 8/1 - White with a slight reddish (more towards pinkish-orange) hue. Very high value and very low chroma. Often reflects parent material color or deposits like carbonates with minor iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Influence", "Possible Carbonate Presence", "Slight Iron Staining"]
  },
  "7.5R 8/2": {
    "name": "Pinkish White",
    "hex": "#E7D6D2",
    "description": "7.5R 8/2 - Pinkish White. Very high value and low chroma. Suggests a predominantly light-colored mineral base with a subtle influence of reddish-yellow iron compounds.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Minor Iron Oxides", "Good Aeration"]
  },
  "7.5R 8/3": {
    "name": "Pink",
    "hex": "#E3CAC6",
    "description": "7.5R 8/3 - Pink. High value, low to moderate chroma. Light reddish-pink color, often associated with soils with finely disseminated hematite having a slightly yellower tint.",
    "properties": ["Light Colored Minerals", "Disseminated Hematite (slight yellow tint)", "Low Organic Matter", "Well-Drained"]
  },
  "7.5R 8/4": {
    "name": "Pink",
    "hex": "#DFBEC0",
    "description": "7.5R 8/4 - Pink. High value, moderate chroma. More pronounced pink color with a warm undertone, indicating a greater presence of hematite in a light-colored matrix.",
    "properties": ["Moderate Hematite", "Well-Aerated", "Good Drainage", "Low Organic Matter"]
  },
  "7.5R 7/1": {
    "name": "Light Gray",
    "hex": "#D1C9C7",
    "description": "7.5R 7/1 - Light Gray with a slight reddish-pink hue. High value, very low chroma. Predominantly gray with a hint of warm red.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Warm Red Iron Influence", "Parent Material Color"]
  },
  "7.5R 7/2": {
    "name": "Pinkish Gray",
    "hex": "#CABAAE",
    "description": "7.5R 7/2 - Pinkish Gray. High value, low chroma. A muted reddish-gray with a warm tone, indicating a mix of light-colored minerals with some hematite.",
    "properties": ["Mixed Mineralogy", "Some Hematite", "Low Organic Matter", "Generally Well-Drained"]
  },
  "7.5R 7/3": {
    "name": "Pink",
    "hex": "#C4ACA9",
    "description": "7.5R 7/3 - Pink. High value, low-moderate chroma. A pale but distinct warm pink, suggesting a fair amount of finely distributed hematite.",
    "properties": ["Finely Distributed Hematite", "Good Aeration", "Low Organic Matter"]
  },
  "7.5R 7/4": {
    "name": "Pink",
    "hex": "#BF9F9B",
    "description": "7.5R 7/4 - Pink. High value, moderate chroma. Clear warm pink color, indicating visible hematite content within a lighter soil matrix.",
    "properties": ["Visible Hematite", "Well-Drained", "Oxidizing Environment"]
  },
  "7.5R 7/6": {
    "name": "Light Red",
    "hex": "#B88B85",
    "description": "7.5R 7/6 - Light Red. High value, moderate to high chroma. A brighter, light red color with a slightly yellower undertone than 5R equivalents, suggesting significant hematite.",
    "properties": ["Significant Hematite", "Good Aeration and Drainage", "May be from Red Parent Materials"]
  },
  "7.5R 7/8": {
    "name": "Light Red",
    "hex": "#B1786F",
    "description": "7.5R 7/8 - Light Red. High value, high chroma. Strong light red color (warm), indicating abundant hematite.",
    "properties": ["Abundant Hematite", "Well-Oxidized", "Often in Weathered Soils or Red Sediments"]
  },
  "7.5R 6/1": {
    "name": "Gray",
    "hex": "#B2AEA9",
    "description": "7.5R 6/1 - Gray with a warm reddish hue. Moderate value, very low chroma. A neutral gray with a slight warm red tint.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Slight Warm Reddish Tinge"]
  },
  "7.5R 6/2": {
    "name": "Pinkish Gray",
    "hex": "#AA9E9A",
    "description": "7.5R 6/2 - Pinkish Gray. Moderate value, low chroma. A muted reddish color (warm tone) mixed with gray, indicating some iron oxidation but not dominant.",
    "properties": ["Some Oxidized Iron", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "7.5R 6/3": {
    "name": "Light Reddish Brown",
    "hex": "#A38E88",
    "description": "7.5R 6/3 - Light Reddish Brown. Moderate value, low-moderate chroma. Soils where red iron oxides (with a slightly yellower hint) are present but mixed with other minerals.",
    "properties": ["Mixed Iron Oxides and Minerals", "Fair Drainage", "Moderate Organic Matter possible"]
  },
  "7.5R 6/4": {
    "name": "Light Reddish Brown",
    "hex": "#9C7F79",
    "description": "7.5R 6/4 - Light Reddish Brown. Moderate value, moderate chroma. A more distinct reddish brown with a warm character, indicating a good presence of hematite.",
    "properties": ["Clear Hematite Presence", "Well-Aerated", "Good Drainage"]
  },
  "7.5R 6/6": {
    "name": "Red",
    "hex": "#946A62",
    "description": "7.5R 6/6 - Red. Moderate value, moderate to high chroma. A clear red color (warm tone), typical of soils with significant hematite accumulation.",
    "properties": ["Significant Hematite", "Well-Drained", "Often in B horizons of Alfisols or Ultisols"]
  },
  "7.5R 6/8": {
    "name": "Red",
    "hex": "#8B564D",
    "description": "7.5R 6/8 - Red. Moderate value, high chroma. A strong red color (warm), indicating high concentration of hematite, common in tropical or highly weathered soils.",
    "properties": ["High Hematite Concentration", "Intense Weathering", "Well-Oxidized Environment"]
  },
  "7.5R 5/1": {
    "name": "Dark Gray",
    "hex": "#817E7A",
    "description": "7.5R 5/1 - Dark Gray with a warm reddish hue. Moderate to low value, very low chroma. Dark color could be from some organic matter or parent rock, with a subtle warm red hint.",
    "properties": ["Some Organic Matter or Dark Minerals", "Slight Warm Reddish Influence", "Variable Drainage"]
  },
  "7.5R 5/2": {
    "name": "Weak Red",
    "hex": "#7D716D",
    "description": "7.5R 5/2 - Weak Red. Moderate to low value, low chroma. A dull, weak red with a warm tone, suggesting limited hematite or masking by other components.",
    "properties": ["Limited Hematite", "May have Organic Mixing", "Moderate Aeration"]
  },
  "7.5R 5/3": {
    "name": "Reddish Brown",
    "hex": "#7B6661",
    "description": "7.5R 5/3 - Reddish Brown. Moderate to low value, low-moderate chroma. Common color for soils with moderate hematite content (warm tone) and some organic matter.",
    "properties": ["Moderate Hematite", "Some Organic Matter", "Generally Good Structure"]
  },
  "7.5R 5/4": {
    "name": "Reddish Brown",
    "hex": "#795D57",
    "description": "7.5R 5/4 - Reddish Brown. Moderate to low value, moderate chroma. A distinct reddish-brown (warm), indicative of well-aerated conditions and iron oxidation.",
    "properties": ["Well-Aerated", "Iron Oxidation (Hematite)", "Good Agricultural Potential often"]
  },
  "7.5R 5/6": {
    "name": "Red",
    "hex": "#764F48",
    "description": "7.5R 5/6 - Red. Moderate to low value, moderate to high chroma. A clear, strong red (warm), suggesting high amounts of hematite.",
    "properties": ["High Hematite Content", "Good Drainage", "Often Associated with Clay Accumulation"]
  },
  "7.5R 5/8": {
    "name": "Red",
    "hex": "#733F37",
    "description": "7.5R 5/8 - Red. Moderate to low value, high chroma. A very strong, relatively dark red (warm). Indicates very high concentration of hematite.",
    "properties": ["Very High Hematite", "Intensely Oxidized", "Common in Tropical Latosols/Oxisols"]
  },
  "7.5R 4/1": {
    "name": "Dark Reddish Gray",
    "hex": "#615E5C",
    "description": "7.5R 4/1 - Dark Reddish Gray. Low value, very low chroma. Dark color often due to organic matter, with a slight warm reddish hue from iron oxides.",
    "properties": ["Organic Matter Influence", "Subtle Iron Oxide Presence (warm tint)", "Moisture Retentive"]
  },
  "7.5R 4/2": {
    "name": "Dark Reddish Gray",
    "hex": "#5F5451",
    "description": "7.5R 4/2 - Dark Reddish Gray. Low value, low chroma. A dark soil with a muted reddish color (warm), suggesting organic matter mixed with some hematite.",
    "properties": ["Organic Matter Rich", "Some Hematite", "Typically Fertile"]
  },
  "7.5R 4/3": {
    "name": "Dark Reddish Brown",
    "hex": "#5F4D48",
    "description": "7.5R 4/3 - Dark Reddish Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides (warm tone) contribute to the color.",
    "properties": ["Good Organic Content", "Moderate Hematite", "Good Soil Structure"]
  },
  "7.5R 4/4": {
    "name": "Dark Reddish Brown",
    "hex": "#604640",
    "description": "7.5R 4/4 - Dark Reddish Brown. Low value, moderate chroma. A rich, dark reddish-brown (warm) indicating good aeration, iron oxides, and organic matter.",
    "properties": ["Well-Aerated", "Productive Soil", "Balanced Organic and Iron Content"]
  },
  "7.5R 4/6": {
    "name": "Dark Red",
    "hex": "#5F3B34",
    "description": "7.5R 4/6 - Dark Red. Low value, moderate to high chroma. A strong dark red (warm), suggesting high hematite content, possibly with some organic matter.",
    "properties": ["High Hematite", "May have Some Organic Matter", "Good Drainage"]
  },
  "7.5R 4/8": {
    "name": "Dark Red",
    "hex": "#5F2F27",
    "description": "7.5R 4/8 - Dark Red. Low value, high chroma. An intense dark red (warm), indicative of very high hematite concentration.",
    "properties": ["Very High Hematite Concentration", "Strongly Oxidized", "Often in Weathered Parent Materials"]
  },
  "7.5R 3/1": {
    "name": "Dark Gray",
    "hex": "#4B4947",
    "description": "7.5R 3/1 - Dark Gray (almost black) with a warm reddish hue. Very low value, very low chroma. Very dark, organic-rich soil with a subtle hint of underlying reddish minerals.",
    "properties": ["High Organic Matter", "Slight Reddish Mineral Influence (warm)", "Often Poorly Drained if very dark and low chroma"]
  },
  "7.5R 3/2": {
    "name": "Dusky Red",
    "hex": "#4A413E",
    "description": "7.5R 3/2 - Dusky Red. Very low value, low chroma. A very dark, muted red with a warm tone. High organic matter content strongly influences this color, mixed with hematite.",
    "properties": ["Very High Organic Matter", "Hematite Present", "Moisture Retentive", "Fertile"]
  },
  "7.5R 3/3": {
    "name": "Dusky Red",
    "hex": "#4C3D39",
    "description": "7.5R 3/3 - Dusky Red. Very low value, low-moderate chroma. Dark reddish color (warm), suggesting a balance of organic matter and significant iron oxide (hematite).",
    "properties": ["Significant Organic Matter and Hematite", "Good Fertility", "Moist Conditions Possible"]
  },
  "7.5R 3/4": {
    "name": "Dusky Red",
    "hex": "#4E3731",
    "description": "7.5R 3/4 - Dusky Red. Very low value, moderate chroma. A darker red (warm), less masked by organic matter than /2 or /3, but still dark. Strong hematite presence.",
    "properties": ["Strong Hematite Presence", "Moderate Organic Matter", "Well-Aerated for a dark soil"]
  },
  "7.5R 3/6": {
    "name": "Dark Red",
    "hex": "#502C24",
    "description": "7.5R 3/6 - Dark Red. Very low value, moderate to high chroma. A deep, dark red color (warm), indicating very high hematite content with less organic matter influence than lower chromas.",
    "properties": ["Very High Hematite", "Good Drainage if structure allows", "Parent Material can be iron-rich rock"]
  },
  "7.5R 3/8": {
    "name": "Dark Red",
    "hex": "#512216",
    "description": "7.5R 3/8 - Dark Red. Very low value, high chroma. An intense, very dark red (warm); nearly pure hematite coloration at a low value.",
    "properties": ["Concentrated Hematite", "Strongly Oxidized Environment", "Can be Hard and Dense"]
  },
  "7.5R 2.5/1": {
    "name": "Black",
    "hex": "#3C3B3A",
    "description": "7.5R 2.5/1 - Black with a warm reddish hue. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or charcoal with a very faint hint of warm red.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal Possible", "Often Waterlogged", "Very Poorly Drained"]
  },
  "7.5R 2.5/2": {
    "name": "Reddish Black",
    "hex": "#3B3331",
    "description": "7.5R 2.5/2 - Reddish Black. Extremely low value, low chroma. Very dark soil where organic matter dominates, but with a noticeable warm reddish influence from iron oxides.",
    "properties": ["Dominant Organic Matter", "Noticeable Hematite (warm tint)", "Poor Drainage Common", "High Fertility when drained"]
  },
  "7.5R 2.5/3": {
    "name": "Reddish Black",
    "hex": "#3E2F2C",
    "description": "7.5R 2.5/3 - Reddish Black. Extremely low value, low-moderate chroma. A very dark soil with a more apparent warm reddish character than 2.5/2.",
    "properties": ["High Organic Matter", "Clearer Warm Reddish Tinge from Hematite", "Moist Environment"]
  },
  "7.5R 2.5/4": {
    "name": "Dark Dusky Red",
    "hex": "#402A25",
    "description": "7.5R 2.5/4 - Dark Dusky Red. Extremely low value, moderate chroma. Darkest expression of warm red before becoming black, significant hematite under very low value conditions.",
    "properties": ["Significant Hematite in Dark Matrix (warm tone)", "Some Organic Matter", "Low Light Reflectance"]
  },
  "10R 8/1": {
    "name": "White",
    "hex": "#EBE3E0",
    "description": "10R 8/1 - White with a faint reddish (pinkish) cast. Very high value and very low chroma. Often reflects parent material color, like light-colored sediments or volcanic ash with minimal iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated", "Minimal Iron Staining", "Possibly Calcareous"]
  },
  "10R 8/2": {
    "name": "Pinkish White",
    "hex": "#E8D7D3",
    "description": "10R 8/2 - Pinkish White. Very high value and low chroma. A very light color with a more noticeable pinkish tint due to disseminated iron oxides (hematite) in a light matrix.",
    "properties": ["Low Organic Matter", "Disseminated Hematite", "Well-Aerated", "Light Mineral Base"]
  },
  "10R 8/3": {
    "name": "Pink",
    "hex": "#E4CBC7",
    "description": "10R 8/3 - Pink. High value, low to moderate chroma. A pale but definite pink, indicating a greater presence of finely distributed hematite.",
    "properties": ["Finely Distributed Hematite", "Good Aeration", "Low Organic Matter", "Often from Weathered Felsic Rocks"]
  },
  "10R 8/4": {
    "name": "Pink",
    "hex": "#E0BFBA",
    "description": "10R 8/4 - Pink. High value, moderate chroma. A clearer pink color, showing a more significant concentration of hematite within the light-colored soil material.",
    "properties": ["Moderate Hematite Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "10R 7/1": {
    "name": "Light Gray",
    "hex": "#D2CAC8",
    "description": "10R 7/1 - Light Gray with a reddish (pinkish) hue. High value, very low chroma. Predominantly gray but with a discernible pinkish tint from slight iron oxide influence.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Hematite Influence", "Parent Material Color often visible"]
  },
  "10R 7/2": {
    "name": "Pinkish Gray",
    "hex": "#CBBBC0",
    "description": "10R 7/2 - Pinkish Gray. High value, low chroma. A muted reddish-gray, indicating a mixture of light-colored minerals with some finely divided hematite.",
    "properties": ["Mixed Mineralogy", "Some Hematite", "Low Organic Matter", "Generally Well-Drained"]
  },
  "10R 7/3": {
    "name": "Pink",
    "hex": "#C5ACA9",
    "description": "10R 7/3 - Pink. High value, low-moderate chroma. A pale pink color, characteristic of soils with a modest amount of hematite.",
    "properties": ["Modest Hematite Content", "Good Aeration", "Low Organic Matter"]
  },
  "10R 7/4": {
    "name": "Pink",
    "hex": "#C09F9A",
    "description": "10R 7/4 - Pink. High value, moderate chroma. A distinct pink, showing a clear presence of hematite in a well-aerated environment.",
    "properties": ["Clear Hematite Presence", "Well-Drained", "Oxidizing Environment"]
  },
  "10R 7/6": {
    "name": "Light Red",
    "hex": "#B98B83",
    "description": "10R 7/6 - Light Red. High value, moderate to high chroma. A brighter, light red color indicating a more substantial accumulation of hematite.",
    "properties": ["Substantial Hematite", "Good Aeration and Drainage", "Often from Reddish Parent Materials or Weathering"]
  },
  "10R 7/8": {
    "name": "Light Red",
    "hex": "#B2776D",
    "description": "10R 7/8 - Light Red. High value, high chroma. A strong and clear light red, signifying abundant and relatively pure hematite.",
    "properties": ["Abundant Hematite", "Well-Oxidized", "May indicate highly weathered conditions or specific parent rocks"]
  },
  "10R 6/1": {
    "name": "Gray",
    "hex": "#B3ADA9",
    "description": "10R 6/1 - Gray with a reddish (pinkish) hue. Moderate value, very low chroma. A neutral gray appearance with a subtle pinkish influence, often from parent material.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Pinkish Tinge (Hematite)"]
  },
  "10R 6/2": {
    "name": "Pinkish Gray",
    "hex": "#AB9E9B",
    "description": "10R 6/2 - Pinkish Gray. Moderate value, low chroma. A muted reddish color mixed with gray, indicating some iron oxidation but not dominant over the base mineral color.",
    "properties": ["Some Oxidized Iron (Hematite)", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "10R 6/3": {
    "name": "Pale Red",
    "hex": "#A48F89",
    "description": "10R 6/3 - Pale Red. Moderate value, low-moderate chroma. A lighter, less saturated red, often seen where hematite is present but mixed with other lighter minerals.",
    "properties": ["Mixed Hematite and Light Minerals", "Fair Drainage", "May have some organic influence dulling the red"]
  },
  "10R 6/4": {
    "name": "Pale Red",
    "hex": "#9D7F79",
    "description": "10R 6/4 - Pale Red. Moderate value, moderate chroma. A more distinct but still light red, indicating a good presence of hematite in a relatively light matrix.",
    "properties": ["Good Hematite Presence", "Well-Aerated", "Moderate Drainage"]
  },
  "10R 6/6": {
    "name": "Red",
    "hex": "#956B62",
    "description": "10R 6/6 - Red. Moderate value, moderate to high chroma. A clear red color, typical of soils with significant hematite accumulation, often in subsoil horizons.",
    "properties": ["Significant Hematite Accumulation", "Well-Drained", "Often in B horizons (argillic, kandic)"]
  },
  "10R 6/8": {
    "name": "Red",
    "hex": "#8E574D",
    "description": "10R 6/8 - Red. Moderate value, high chroma. A strong red color, indicating a high concentration of hematite, common in highly weathered soils or those from iron-rich parent materials.",
    "properties": ["High Hematite Concentration", "Intense Weathering Product", "Well-Oxidized Environment"]
  },
  "10R 5/1": {
    "name": "Reddish Gray",
    "hex": "#817E7B",
    "description": "10R 5/1 - Reddish Gray. Moderate to low value, very low chroma. A darker gray with a discernible reddish hue, suggesting some organic matter or dark minerals with slight iron oxide influence.",
    "properties": ["Some Organic Matter or Dark Minerals", "Slight Hematite Influence", "Variable Drainage Conditions"]
  },
  "10R 5/2": {
    "name": "Weak Red",
    "hex": "#7D716E",
    "description": "10R 5/2 - Weak Red. Moderate to low value, low chroma. A dull, weak red, suggesting limited hematite content or significant masking by organic matter or other minerals.",
    "properties": ["Limited Hematite", "Organic Matter Masking Possible", "Moderate Aeration"]
  },
  "10R 5/3": {
    "name": "Weak Red",
    "hex": "#7B6661",
    "description": "10R 5/3 - Weak Red. Moderate to low value, low-moderate chroma. A reddish color that is not very vivid, often found in soils with moderate hematite and some organic incorporation.",
    "properties": ["Moderate Hematite", "Some Organic Incorporation", "Generally Good Soil Structure"]
  },
  "10R 5/4": {
    "name": "Reddish Brown",
    "hex": "#795D57",
    "description": "10R 5/4 - Reddish Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good iron (hematite) presence, less red than higher chromas.",
    "properties": ["Well-Aerated", "Good Hematite Presence", "Often Good Agricultural Soils"]
  },
  "10R 5/6": {
    "name": "Red",
    "hex": "#764F48",
    "description": "10R 5/6 - Red. Moderate to low value, moderate to high chroma. A clear, strong red, indicating high amounts of hematite and good oxidation.",
    "properties": ["High Hematite Content", "Good Drainage and Aeration", "Often found in well-developed soil profiles"]
  },
  "10R 5/8": {
    "name": "Red",
    "hex": "#733F37",
    "description": "10R 5/8 - Red. Moderate to low value, high chroma. A very strong, relatively dark red. Indicates a very high concentration of hematite and strong weathering.",
    "properties": ["Very High Hematite Content", "Intensely Oxidized", "Common in tropical or subtropical regions (e.g., Oxisols)"]
  },
  "10R 4/1": {
    "name": "Dark Reddish Gray",
    "hex": "#615E5C",
    "description": "10R 4/1 - Dark Reddish Gray. Low value, very low chroma. Dark color, often due to significant organic matter content, with a slight reddish hue from underlying iron oxides.",
    "properties": ["Significant Organic Matter Influence", "Subtle Hematite Presence", "Moisture Retentive", "Often fertile A-horizons"]
  },
  "10R 4/2": {
    "name": "Dark Reddish Gray",
    "hex": "#5F5451",
    "description": "10R 4/2 - Dark Reddish Gray. Low value, low chroma. A dark soil with a muted reddish color, suggesting a mix of considerable organic matter and some hematite.",
    "properties": ["Considerable Organic Matter", "Some Hematite", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "10R 4/3": {
    "name": "Dark Reddish Brown",
    "hex": "#5F4D48",
    "description": "10R 4/3 - Dark Reddish Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides (hematite) both contribute significantly to the color.",
    "properties": ["Good Organic Content", "Moderate Hematite", "Good Soil Structure", "Productive for Agriculture"]
  },
  "10R 4/4": {
    "name": "Dark Reddish Brown",
    "hex": "#604640",
    "description": "10R 4/4 - Dark Reddish Brown. Low value, moderate chroma. A rich, dark reddish-brown indicating good aeration, significant iron oxides, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Iron Content"]
  },
  "10R 4/6": {
    "name": "Dark Red",
    "hex": "#5F3B34",
    "description": "10R 4/6 - Dark Red. Low value, moderate to high chroma. A strong dark red, suggesting high hematite content, which may be accompanied by some organic matter.",
    "properties": ["High Hematite Content", "May have Some Organic Matter", "Good Drainage", "Often in stable, well-weathered landscapes"]
  },
  "10R 4/8": {
    "name": "Dark Red",
    "hex": "#5F2F27",
    "description": "10R 4/8 - Dark Red. Low value, high chroma. An intense dark red, indicative of a very high concentration of hematite, possibly in ironstone formations or highly ferruginous soils.",
    "properties": ["Very High Hematite Concentration", "Strongly Oxidized Conditions", "May be associated with Plinthite or Laterite"]
  },
  "10R 3/1": {
    "name": "Dark Gray",
    "hex": "#4B4947",
    "description": "10R 3/1 - Dark Gray (almost black) with a reddish hue. Very low value, very low chroma. Very dark, typically organic-rich soil with only a subtle hint of underlying reddish minerals (hematite).",
    "properties": ["High Organic Matter Content", "Slight Reddish Mineral Influence", "Often in moist to wet conditions", "Fertile when drained"]
  },
  "10R 3/2": {
    "name": "Dusky Red",
    "hex": "#4A413E",
    "description": "10R 3/2 - Dusky Red. Very low value, low chroma. A very dark, muted red. High organic matter content strongly influences this color, mixed with hematite.",
    "properties": ["Very High Organic Matter", "Hematite Present but Masked", "Moisture Retentive", "Typically Fertile"]
  },
  "10R 3/3": {
    "name": "Dusky Red",
    "hex": "#4C3D39",
    "description": "10R 3/3 - Dusky Red. Very low value, low-moderate chroma. Dark reddish color, suggesting a balance of significant organic matter and notable iron oxide (hematite) content.",
    "properties": ["Significant Organic Matter and Hematite", "Good Fertility", "Moist Environment common"]
  },
  "10R 3/4": {
    "name": "Dusky Red",
    "hex": "#4E3731",
    "description": "10R 3/4 - Dusky Red. Very low value, moderate chroma. A darker red, where hematite is more expressed than in lower chromas but still within a dark matrix, likely organic-influenced.",
    "properties": ["Strong Hematite Presence in Dark Matrix", "Moderate Organic Matter", "Good Aeration for a dark soil"]
  },
  "10R 3/6": {
    "name": "Dark Red",
    "hex": "#502C24",
    "description": "10R 3/6 - Dark Red. Very low value, moderate to high chroma. A deep, dark red color, indicating very high hematite content with less overwhelming organic matter influence than at lower chromas.",
    "properties": ["Very High Hematite Content", "Well-Drained if structure permits", "Can indicate iron-rich parent material or concretions"]
  },
  "10R 2.5/1": {
    "name": "Black",
    "hex": "#3C3B3A",
    "description": "10R 2.5/1 - Black with a faint reddish hue. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or material like charcoal, with the faintest hint of red.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained"]
  },
  "10R 2.5/2": {
    "name": "Reddish Black",
    "hex": "#3B3331",
    "description": "10R 2.5/2 - Reddish Black. Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable reddish influence from finely disseminated hematite.",
    "properties": ["Dominant Organic Matter", "Noticeable Hematite Influence", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "10R 2.5/3": {
    "name": "Reddish Black",
    "hex": "#3E2F2C",
    "description": "10R 2.5/3 - Reddish Black. Extremely low value, low-moderate chroma. A very dark soil with a more apparent reddish character from hematite compared to 2.5/2, still organic-rich.",
    "properties": ["High Organic Matter", "Clearer Reddish Tinge from Hematite", "Moist to Wet Environment", "Often acidic"]
  },
  "10R 2.5/4": {
    "name": "Dark Dusky Red",
    "hex": "#402A25",
    "description": "10R 2.5/4 - Dark Dusky Red. Extremely low value, moderate chroma. The darkest expression of a distinct red, indicating significant hematite content even under very low value (dark) conditions, likely mixed with organic material.",
    "properties": ["Significant Hematite in a Very Dark Matrix", "Organic Material Mixture", "Low Light Reflectance", "Specific formation conditions"]
  },
}


class MunsellClassifier:{
  "2.5YR 8/1": {
    "name": "White",
    "hex": "#EBE3E0",
    "description": "2.5YR 8/1 - White with a very faint reddish-yellow cast. Very high value and very low chroma. Often reflects light-colored parent material with minimal iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated", "Minimal Iron Staining (Reddish-Yellow)", "Possibly Calcareous or Siliceous"]
  },
  "2.5YR 8/2": {
    "name": "Pinkish White",
    "hex": "#E7D7D2",
    "description": "2.5YR 8/2 - Pinkish White (with a yellowish tint). Very high value and low chroma. A very light color with a noticeable warm pinkish-yellow tint from disseminated iron oxides.",
    "properties": ["Low Organic Matter", "Disseminated Iron Oxides", "Well-Aerated", "Light Mineral Base"]
  },
  "2.5YR 8/3": {
    "name": "Pink",
    "hex": "#E4CBC6",
    "description": "2.5YR 8/3 - Pink (warm, slightly yellowish). High value, low to moderate chroma. A pale but definite warm pink, indicating a greater presence of finely distributed iron oxides.",
    "properties": ["Finely Distributed Iron Oxides", "Good Aeration", "Low Organic Matter", "Often from Weathered Granitic or Sandy Parent Materials"]
  },
  "2.5YR 8/4": {
    "name": "Pink",
    "hex": "#E0BFBA",
    "description": "2.5YR 8/4 - Pink (warm). High value, moderate chroma. A clearer warm pink color, showing a more significant concentration of iron oxides within the light-colored soil material.",
    "properties": ["Moderate Iron Oxide Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "2.5YR 7/1": {
    "name": "Light Gray",
    "hex": "#D2CAC8",
    "description": "2.5YR 7/1 - Light Gray with a reddish-yellow hue. High value, very low chroma. Predominantly gray but with a discernible warm pinkish-yellow tint from slight iron oxide influence.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Iron Oxide Influence (Reddish-Yellow)", "Parent Material Color often visible"]
  },
  "2.5YR 7/2": {
    "name": "Pinkish Gray",
    "hex": "#CBBAB6",
    "description": "2.5YR 7/2 - Pinkish Gray (warm). High value, low chroma. A muted reddish-yellow gray, indicating a mixture of light-colored minerals with some finely divided iron oxides.",
    "properties": ["Mixed Mineralogy", "Some Iron Oxides", "Low Organic Matter", "Generally Well-Drained"]
  },
  "2.5YR 7/3": {
    "name": "Light Reddish Brown",
    "hex": "#C4ACA8",
    "description": "2.5YR 7/3 - Light Reddish Brown (pale). High value, low-moderate chroma. A pale reddish-brown, characteristic of soils with a modest amount of mixed iron oxides.",
    "properties": ["Modest Iron Oxide Content", "Good Aeration", "Low Organic Matter"]
  },
  "2.5YR 7/4": {
    "name": "Light Reddish Brown",
    "hex": "#BF9F9A",
    "description": "2.5YR 7/4 - Light Reddish Brown. High value, moderate chroma. A distinct light reddish-brown, showing a clear presence of iron oxides in a well-aerated environment.",
    "properties": ["Clear Iron Oxide Presence", "Well-Drained", "Oxidizing Environment"]
  },
  "2.5YR 7/6": {
    "name": "Reddish Yellow",
    "hex": "#B88B82",
    "description": "2.5YR 7/6 - Reddish Yellow. High value, moderate to high chroma. A brighter, light reddish-yellow indicating a more substantial accumulation of (goethite and some hematite).",
    "properties": ["Substantial Iron Oxides", "Good Aeration and Drainage", "Often from Reddish-Yellow Parent Materials or Weathering"]
  },
  "2.5YR 7/8": {
    "name": "Reddish Yellow",
    "hex": "#B1776B",
    "description": "2.5YR 7/8 - Reddish Yellow. High value, high chroma. A strong and clear light reddish-yellow, signifying abundant and relatively pure mixed iron oxides.",
    "properties": ["Abundant Iron Oxides", "Well-Oxidized", "May indicate highly weathered conditions or specific parent rocks with iron"]
  },
  "2.5YR 6/1": {
    "name": "Gray",
    "hex": "#B3ADA9",
    "description": "2.5YR 6/1 - Gray with a reddish-yellow hue. Moderate value, very low chroma. A neutral gray appearance with a subtle warm pinkish-yellow influence.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Reddish-Yellow Tinge (Iron Oxides)"]
  },
  "2.5YR 6/2": {
    "name": "Light Brownish Gray",
    "hex": "#AB9E9A",
    "description": "2.5YR 6/2 - Light Brownish Gray (with reddish tint). Moderate value, low chroma. A muted reddish-yellow color mixed with gray, indicating some iron oxidation but not dominant.",
    "properties": ["Some Oxidized Iron", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "2.5YR 6/3": {
    "name": "Light Reddish Brown",
    "hex": "#A48F89",
    "description": "2.5YR 6/3 - Light Reddish Brown. Moderate value, low-moderate chroma. A lighter, less saturated reddish-brown, often where iron oxides are present but mixed with other lighter minerals.",
    "properties": ["Mixed Iron Oxides and Light Minerals", "Fair Drainage", "May have some organic influence dulling the color"]
  },
  "2.5YR 6/4": {
    "name": "Light Reddish Brown",
    "hex": "#9D7F78",
    "description": "2.5YR 6/4 - Light Reddish Brown. Moderate value, moderate chroma. A more distinct but still light reddish-brown, indicating a good presence of iron oxides in a relatively light matrix.",
    "properties": ["Good Iron Oxide Presence", "Well-Aerated", "Moderate Drainage"]
  },
  "2.5YR 6/6": {
    "name": "Reddish Yellow",
    "hex": "#956B60",
    "description": "2.5YR 6/6 - Reddish Yellow. Moderate value, moderate to high chroma. A clear reddish-yellow, typical of soils with significant mixed iron oxide accumulation.",
    "properties": ["Significant Iron Oxide Accumulation", "Well-Drained", "Often in B horizons (argillic, kandic) with good aeration"]
  },
  "2.5YR 6/8": {
    "name": "Yellowish Red",
    "hex": "#8E574B",
    "description": "2.5YR 6/8 - Yellowish Red. Moderate value, high chroma. A strong yellowish-red, indicating a high concentration of (hematite and goethite), common in highly weathered soils.",
    "properties": ["High Iron Oxide Concentration", "Intense Weathering Product", "Well-Oxidized Environment"]
  },
  "2.5YR 5/1": {
    "name": "Reddish Gray",
    "hex": "#817E7A",
    "description": "2.5YR 5/1 - Reddish Gray. Moderate to low value, very low chroma. A darker gray with a discernible reddish-yellow hue, suggesting some organic matter or dark minerals with slight iron oxide influence.",
    "properties": ["Some Organic Matter or Dark Minerals", "Slight Iron Oxide Influence (Reddish-Yellow)", "Variable Drainage Conditions"]
  },
  "2.5YR 5/2": {
    "name": "Weak Red",
    "hex": "#7D716D",
    "description": "2.5YR 5/2 - Weak Red (with yellowish cast). Moderate to low value, low chroma. A dull, weak reddish-yellow, suggesting limited iron oxide content or significant masking by organic matter or other minerals.",
    "properties": ["Limited Iron Oxides", "Organic Matter Masking Possible", "Moderate Aeration"]
  },
  "2.5YR 5/3": {
    "name": "Reddish Brown",
    "hex": "#7B6660",
    "description": "2.5YR 5/3 - Reddish Brown. Moderate to low value, low-moderate chroma. A common soil color, often found in soils with moderate iron oxides and some organic incorporation.",
    "properties": ["Moderate Iron Oxides", "Some Organic Incorporation", "Generally Good Soil Structure"]
  },
  "2.5YR 5/4": {
    "name": "Reddish Brown",
    "hex": "#795D56",
    "description": "2.5YR 5/4 - Reddish Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good iron oxide presence.",
    "properties": ["Well-Aerated", "Good Iron Oxide Presence", "Often Good Agricultural Soils"]
  },
  "2.5YR 5/6": {
    "name": "Yellowish Red",
    "hex": "#764F46",
    "description": "2.5YR 5/6 - Yellowish Red. Moderate to low value, moderate to high chroma. A clear, strong yellowish-red, indicating high amounts of mixed iron oxides and good oxidation.",
    "properties": ["High Iron Oxide Content", "Good Drainage and Aeration", "Often found in well-developed soil profiles"]
  },
  "2.5YR 5/8": {
    "name": "Yellowish Red",
    "hex": "#733F35",
    "description": "2.5YR 5/8 - Yellowish Red. Moderate to low value, high chroma. A very strong, relatively dark yellowish-red. Indicates a very high concentration of iron oxides and strong weathering.",
    "properties": ["Very High Iron Oxide Content", "Intensely Oxidized", "Common in tropical or subtropical regions (e.g., Oxisols, Ultisols)"]
  },
  "2.5YR 4/1": {
    "name": "Dark Reddish Gray",
    "hex": "#615E5C",
    "description": "2.5YR 4/1 - Dark Reddish Gray. Low value, very low chroma. Dark color, often due to significant organic matter content, with a slight reddish-yellow hue from underlying iron oxides.",
    "properties": ["Significant Organic Matter Influence", "Subtle Iron Oxide Presence (Reddish-Yellow)", "Moisture Retentive", "Often fertile A-horizons"]
  },
  "2.5YR 4/2": {
    "name": "Dark Grayish Brown",
    "hex": "#5F5451",
    "description": "2.5YR 4/2 - Dark Grayish Brown (with reddish tint). Low value, low chroma. A dark soil with a muted reddish-yellow color, suggesting a mix of considerable organic matter and some iron oxides.",
    "properties": ["Considerable Organic Matter", "Some Iron Oxides", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "2.5YR 4/3": {
    "name": "Dark Reddish Brown",
    "hex": "#5F4D48",
    "description": "2.5YR 4/3 - Dark Reddish Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides both contribute significantly to the color.",
    "properties": ["Good Organic Content", "Moderate Iron Oxides", "Good Soil Structure", "Productive for Agriculture"]
  },
  "2.5YR 4/4": {
    "name": "Dark Reddish Brown",
    "hex": "#60463F",
    "description": "2.5YR 4/4 - Dark Reddish Brown. Low value, moderate chroma. A rich, dark reddish-brown indicating good aeration, significant iron oxides, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Iron Content"]
  },
  "2.5YR 4/6": {
    "name": "Dark Red",
    "hex": "#5F3B32",
    "description": "2.5YR 4/6 - Dark Red. Low value, moderate to high chroma. A strong dark red (more red than yellow at this point), suggesting high iron oxide (hematite more prominent) content, possibly with some organic matter.",
    "properties": ["High Iron Oxide Content (more Hematite)", "May have Some Organic Matter", "Good Drainage", "Often in stable, well-weathered landscapes"]
  },
  "2.5YR 4/8": {
    "name": "Dark Red",
    "hex": "#5F2F25",
    "description": "2.5YR 4/8 - Dark Red. Low value, high chroma. An intense dark red, indicative of a very high concentration of hematite-rich iron oxides.",
    "properties": ["Very High Hematite-Rich Iron Oxides", "Strongly Oxidized Conditions", "May be associated with Plinthite or Laterite formations"]
  },
  "2.5YR 3/1": {
    "name": "Very Dusky Red",
    "hex": "#4B4947",
    "description": "2.5YR 3/1 - Very Dusky Red (almost black). Very low value, very low chroma. Very dark, typically organic-rich soil with only a subtle hint of underlying reddish-yellow minerals.",
    "properties": ["High Organic Matter Content", "Slight Reddish-Yellow Mineral Influence", "Often in moist to wet conditions", "Fertile when drained"]
  },
  "2.5YR 3/2": {
    "name": "Very Dusky Red",
    "hex": "#4A413E",
    "description": "2.5YR 3/2 - Very Dusky Red. Very low value, low chroma. A very dark, muted reddish-yellow. High organic matter content strongly influences this color, mixed with iron oxides.",
    "properties": ["Very High Organic Matter", "Iron Oxides Present but Masked", "Moisture Retentive", "Typically Fertile"]
  },
  "2.5YR 3/3": {
    "name": "Dark Brown",
    "hex": "#4C3D38",
    "description": "2.5YR 3/3 - Dark Brown (with reddish cast). Very low value, low-moderate chroma. Dark reddish-yellow brown color, suggesting a balance of significant organic matter and notable iron oxide content.",
    "properties": ["Significant Organic Matter and Iron Oxides", "Good Fertility", "Moist Environment common"]
  },
  "2.5YR 3/4": {
    "name": "Dark Brown",
    "hex": "#4E3730",
    "description": "2.5YR 3/4 - Dark Brown (strong reddish cast). Very low value, moderate chroma. A darker reddish-yellow brown, where iron oxides are more expressed than in lower chromas but still within a dark matrix, likely organic-influenced.",
    "properties": ["Strong Iron Oxide Presence in Dark Matrix", "Moderate Organic Matter", "Good Aeration for a dark soil"]
  },
  "2.5YR 3/6": {
    "name": "Dark Red",
    "hex": "#502C22",
    "description": "2.5YR 3/6 - Dark Red. Very low value, moderate to high chroma. A deep, dark red color, indicating very high hematite-rich iron oxide content with less overwhelming organic matter influence than at lower chromas.",
    "properties": ["Very High Hematite-Rich Iron Oxides", "Well-Drained if structure permits", "Can indicate iron-rich parent material or concretions"]
  },
  "2.5YR 2.5/1": {
    "name": "Black",
    "hex": "#3C3B3A",
    "description": "2.5YR 2.5/1 - Black with a faint reddish-yellow hue. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or material like charcoal, with the faintest hint of reddish-yellow.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained"]
  },
  "2.5YR 2.5/2": {
    "name": "Reddish Black",
    "hex": "#3B3330",
    "description": "2.5YR 2.5/2 - Reddish Black. Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable reddish-yellow influence from finely disseminated iron oxides.",
    "properties": ["Dominant Organic Matter", "Noticeable Iron Oxide Influence (Reddish-Yellow)", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "2.5YR 2.5/3": {
    "name": "Reddish Black",
    "hex": "#3E2F2B",
    "description": "2.5YR 2.5/3 - Reddish Black. Extremely low value, low-moderate chroma. A very dark soil with a more apparent reddish-yellow character from iron oxides compared to 2.5/2, still organic-rich.",
    "properties": ["High Organic Matter", "Clearer Reddish-Yellow Tinge from Iron Oxides", "Moist to Wet Environment", "Often acidic"]
  },
  "2.5YR 2.5/4": {
    "name": "Dark Reddish Brown",
    "hex": "#402A24",
    "description": "2.5YR 2.5/4 - Dark Reddish Brown. Extremely low value, moderate chroma. The darkest expression of a distinct reddish-yellow brown, indicating significant iron oxide content even under very low value (dark) conditions, likely mixed with organic material.",
    "properties": ["Significant Iron Oxides in a Very Dark Matrix", "Organic Material Mixture", "Low Light Reflectance", "Specific formation conditions"]
  },
  "2.5YR 8/1": {
    "name": "White",
    "hex": "#EBE3E0",
    "description": "2.5YR 8/1 - White with a very faint reddish-yellow cast. Very high value and very low chroma. Often reflects light-colored parent material with minimal iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated", "Minimal Iron Staining (Reddish-Yellow)", "Possibly Calcareous or Siliceous"]
  },
  "2.5YR 8/2": {
    "name": "Pinkish White",
    "hex": "#E7D7D2",
    "description": "2.5YR 8/2 - Pinkish White (with a yellowish tint). Very high value and low chroma. A very light color with a noticeable warm pinkish-yellow tint from disseminated iron oxides.",
    "properties": ["Low Organic Matter", "Disseminated Iron Oxides", "Well-Aerated", "Light Mineral Base"]
  },
  "2.5YR 8/3": {
    "name": "Pink",
    "hex": "#E4CBC6",
    "description": "2.5YR 8/3 - Pink (warm, slightly yellowish). High value, low to moderate chroma. A pale but definite warm pink, indicating a greater presence of finely distributed iron oxides.",
    "properties": ["Finely Distributed Iron Oxides", "Good Aeration", "Low Organic Matter", "Often from Weathered Granitic or Sandy Parent Materials"]
  },
  "2.5YR 8/4": {
    "name": "Pink",
    "hex": "#E0BFBA",
    "description": "2.5YR 8/4 - Pink (warm). High value, moderate chroma. A clearer warm pink color, showing a more significant concentration of iron oxides within the light-colored soil material.",
    "properties": ["Moderate Iron Oxide Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "2.5YR 7/1": {
    "name": "Light Gray",
    "hex": "#D2CAC8",
    "description": "2.5YR 7/1 - Light Gray with a reddish-yellow hue. High value, very low chroma. Predominantly gray but with a discernible warm pinkish-yellow tint from slight iron oxide influence.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Iron Oxide Influence (Reddish-Yellow)", "Parent Material Color often visible"]
  },
  "2.5YR 7/2": {
    "name": "Pinkish Gray",
    "hex": "#CBBAB6",
    "description": "2.5YR 7/2 - Pinkish Gray (warm). High value, low chroma. A muted reddish-yellow gray, indicating a mixture of light-colored minerals with some finely divided iron oxides.",
    "properties": ["Mixed Mineralogy", "Some Iron Oxides", "Low Organic Matter", "Generally Well-Drained"]
  },
  "2.5YR 7/3": {
    "name": "Light Reddish Brown",
    "hex": "#C4ACA8",
    "description": "2.5YR 7/3 - Light Reddish Brown (pale). High value, low-moderate chroma. A pale reddish-brown, characteristic of soils with a modest amount of mixed iron oxides.",
    "properties": ["Modest Iron Oxide Content", "Good Aeration", "Low Organic Matter"]
  },
  "2.5YR 7/4": {
    "name": "Light Reddish Brown",
    "hex": "#BF9F9A",
    "description": "2.5YR 7/4 - Light Reddish Brown. High value, moderate chroma. A distinct light reddish-brown, showing a clear presence of iron oxides in a well-aerated environment.",
    "properties": ["Clear Iron Oxide Presence", "Well-Drained", "Oxidizing Environment"]
  },
  "2.5YR 7/6": {
    "name": "Reddish Yellow",
    "hex": "#B88B82",
    "description": "2.5YR 7/6 - Reddish Yellow. High value, moderate to high chroma. A brighter, light reddish-yellow indicating a more substantial accumulation of (goethite and some hematite).",
    "properties": ["Substantial Iron Oxides", "Good Aeration and Drainage", "Often from Reddish-Yellow Parent Materials or Weathering"]
  },
  "2.5YR 7/8": {
    "name": "Reddish Yellow",
    "hex": "#B1776B",
    "description": "2.5YR 7/8 - Reddish Yellow. High value, high chroma. A strong and clear light reddish-yellow, signifying abundant and relatively pure mixed iron oxides.",
    "properties": ["Abundant Iron Oxides", "Well-Oxidized", "May indicate highly weathered conditions or specific parent rocks with iron"]
  },
  "2.5YR 6/1": {
    "name": "Gray",
    "hex": "#B3ADA9",
    "description": "2.5YR 6/1 - Gray with a reddish-yellow hue. Moderate value, very low chroma. A neutral gray appearance with a subtle warm pinkish-yellow influence.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Reddish-Yellow Tinge (Iron Oxides)"]
  },
  "2.5YR 6/2": {
    "name": "Light Brownish Gray",
    "hex": "#AB9E9A",
    "description": "2.5YR 6/2 - Light Brownish Gray (with reddish tint). Moderate value, low chroma. A muted reddish-yellow color mixed with gray, indicating some iron oxidation but not dominant.",
    "properties": ["Some Oxidized Iron", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "2.5YR 6/3": {
    "name": "Light Reddish Brown",
    "hex": "#A48F89",
    "description": "2.5YR 6/3 - Light Reddish Brown. Moderate value, low-moderate chroma. A lighter, less saturated reddish-brown, often where iron oxides are present but mixed with other lighter minerals.",
    "properties": ["Mixed Iron Oxides and Light Minerals", "Fair Drainage", "May have some organic influence dulling the color"]
  },
  "2.5YR 6/4": {
    "name": "Light Reddish Brown",
    "hex": "#9D7F78",
    "description": "2.5YR 6/4 - Light Reddish Brown. Moderate value, moderate chroma. A more distinct but still light reddish-brown, indicating a good presence of iron oxides in a relatively light matrix.",
    "properties": ["Good Iron Oxide Presence", "Well-Aerated", "Moderate Drainage"]
  },
  "2.5YR 6/6": {
    "name": "Reddish Yellow",
    "hex": "#956B60",
    "description": "2.5YR 6/6 - Reddish Yellow. Moderate value, moderate to high chroma. A clear reddish-yellow, typical of soils with significant mixed iron oxide accumulation.",
    "properties": ["Significant Iron Oxide Accumulation", "Well-Drained", "Often in B horizons (argillic, kandic) with good aeration"]
  },
  "2.5YR 6/8": {
    "name": "Yellowish Red",
    "hex": "#8E574B",
    "description": "2.5YR 6/8 - Yellowish Red. Moderate value, high chroma. A strong yellowish-red, indicating a high concentration of (hematite and goethite), common in highly weathered soils.",
    "properties": ["High Iron Oxide Concentration", "Intense Weathering Product", "Well-Oxidized Environment"]
  },
  "2.5YR 5/1": {
    "name": "Reddish Gray",
    "hex": "#817E7A",
    "description": "2.5YR 5/1 - Reddish Gray. Moderate to low value, very low chroma. A darker gray with a discernible reddish-yellow hue, suggesting some organic matter or dark minerals with slight iron oxide influence.",
    "properties": ["Some Organic Matter or Dark Minerals", "Slight Iron Oxide Influence (Reddish-Yellow)", "Variable Drainage Conditions"]
  },
  "2.5YR 5/2": {
    "name": "Weak Red",
    "hex": "#7D716D",
    "description": "2.5YR 5/2 - Weak Red (with yellowish cast). Moderate to low value, low chroma. A dull, weak reddish-yellow, suggesting limited iron oxide content or significant masking by organic matter or other minerals.",
    "properties": ["Limited Iron Oxides", "Organic Matter Masking Possible", "Moderate Aeration"]
  },
  "2.5YR 5/3": {
    "name": "Reddish Brown",
    "hex": "#7B6660",
    "description": "2.5YR 5/3 - Reddish Brown. Moderate to low value, low-moderate chroma. A common soil color, often found in soils with moderate iron oxides and some organic incorporation.",
    "properties": ["Moderate Iron Oxides", "Some Organic Incorporation", "Generally Good Soil Structure"]
  },
  "2.5YR 5/4": {
    "name": "Reddish Brown",
    "hex": "#795D56",
    "description": "2.5YR 5/4 - Reddish Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good iron oxide presence.",
    "properties": ["Well-Aerated", "Good Iron Oxide Presence", "Often Good Agricultural Soils"]
  },
  "2.5YR 5/6": {
    "name": "Yellowish Red",
    "hex": "#764F46",
    "description": "2.5YR 5/6 - Yellowish Red. Moderate to low value, moderate to high chroma. A clear, strong yellowish-red, indicating high amounts of mixed iron oxides and good oxidation.",
    "properties": ["High Iron Oxide Content", "Good Drainage and Aeration", "Often found in well-developed soil profiles"]
  },
  "2.5YR 5/8": {
    "name": "Yellowish Red",
    "hex": "#733F35",
    "description": "2.5YR 5/8 - Yellowish Red. Moderate to low value, high chroma. A very strong, relatively dark yellowish-red. Indicates a very high concentration of iron oxides and strong weathering.",
    "properties": ["Very High Iron Oxide Content", "Intensely Oxidized", "Common in tropical or subtropical regions (e.g., Oxisols, Ultisols)"]
  },
  "2.5YR 4/1": {
    "name": "Dark Reddish Gray",
    "hex": "#615E5C",
    "description": "2.5YR 4/1 - Dark Reddish Gray. Low value, very low chroma. Dark color, often due to significant organic matter content, with a slight reddish-yellow hue from underlying iron oxides.",
    "properties": ["Significant Organic Matter Influence", "Subtle Iron Oxide Presence (Reddish-Yellow)", "Moisture Retentive", "Often fertile A-horizons"]
  },
  "2.5YR 4/2": {
    "name": "Dark Grayish Brown",
    "hex": "#5F5451",
    "description": "2.5YR 4/2 - Dark Grayish Brown (with reddish tint). Low value, low chroma. A dark soil with a muted reddish-yellow color, suggesting a mix of considerable organic matter and some iron oxides.",
    "properties": ["Considerable Organic Matter", "Some Iron Oxides", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "2.5YR 4/3": {
    "name": "Dark Reddish Brown",
    "hex": "#5F4D48",
    "description": "2.5YR 4/3 - Dark Reddish Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides both contribute significantly to the color.",
    "properties": ["Good Organic Content", "Moderate Iron Oxides", "Good Soil Structure", "Productive for Agriculture"]
  },
  "2.5YR 4/4": {
    "name": "Dark Reddish Brown",
    "hex": "#60463F",
    "description": "2.5YR 4/4 - Dark Reddish Brown. Low value, moderate chroma. A rich, dark reddish-brown indicating good aeration, significant iron oxides, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Iron Content"]
  },
  "2.5YR 4/6": {
    "name": "Dark Red",
    "hex": "#5F3B32",
    "description": "2.5YR 4/6 - Dark Red. Low value, moderate to high chroma. A strong dark red (more red than yellow at this point), suggesting high iron oxide (hematite more prominent) content, possibly with some organic matter.",
    "properties": ["High Iron Oxide Content (more Hematite)", "May have Some Organic Matter", "Good Drainage", "Often in stable, well-weathered landscapes"]
  },
  "2.5YR 4/8": {
    "name": "Dark Red",
    "hex": "#5F2F25",
    "description": "2.5YR 4/8 - Dark Red. Low value, high chroma. An intense dark red, indicative of a very high concentration of hematite-rich iron oxides.",
    "properties": ["Very High Hematite-Rich Iron Oxides", "Strongly Oxidized Conditions", "May be associated with Plinthite or Laterite formations"]
  },
  "2.5YR 3/1": {
    "name": "Very Dusky Red",
    "hex": "#4B4947",
    "description": "2.5YR 3/1 - Very Dusky Red (almost black). Very low value, very low chroma. Very dark, typically organic-rich soil with only a subtle hint of underlying reddish-yellow minerals.",
    "properties": ["High Organic Matter Content", "Slight Reddish-Yellow Mineral Influence", "Often in moist to wet conditions", "Fertile when drained"]
  },
  "2.5YR 3/2": {
    "name": "Very Dusky Red",
    "hex": "#4A413E",
    "description": "2.5YR 3/2 - Very Dusky Red. Very low value, low chroma. A very dark, muted reddish-yellow. High organic matter content strongly influences this color, mixed with iron oxides.",
    "properties": ["Very High Organic Matter", "Iron Oxides Present but Masked", "Moisture Retentive", "Typically Fertile"]
  },
  "2.5YR 3/3": {
    "name": "Dark Brown",
    "hex": "#4C3D38",
    "description": "2.5YR 3/3 - Dark Brown (with reddish cast). Very low value, low-moderate chroma. Dark reddish-yellow brown color, suggesting a balance of significant organic matter and notable iron oxide content.",
    "properties": ["Significant Organic Matter and Iron Oxides", "Good Fertility", "Moist Environment common"]
  },
  "2.5YR 3/4": {
    "name": "Dark Brown",
    "hex": "#4E3730",
    "description": "2.5YR 3/4 - Dark Brown (strong reddish cast). Very low value, moderate chroma. A darker reddish-yellow brown, where iron oxides are more expressed than in lower chromas but still within a dark matrix, likely organic-influenced.",
    "properties": ["Strong Iron Oxide Presence in Dark Matrix", "Moderate Organic Matter", "Good Aeration for a dark soil"]
  },
  "2.5YR 3/6": {
    "name": "Dark Red",
    "hex": "#502C22",
    "description": "2.5YR 3/6 - Dark Red. Very low value, moderate to high chroma. A deep, dark red color, indicating very high hematite-rich iron oxide content with less overwhelming organic matter influence than at lower chromas.",
    "properties": ["Very High Hematite-Rich Iron Oxides", "Well-Drained if structure permits", "Can indicate iron-rich parent material or concretions"]
  },
  "2.5YR 2.5/1": {
    "name": "Black",
    "hex": "#3C3B3A",
    "description": "2.5YR 2.5/1 - Black with a faint reddish-yellow hue. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or material like charcoal, with the faintest hint of reddish-yellow.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained"]
  },
  "2.5YR 2.5/2": {
    "name": "Reddish Black",
    "hex": "#3B3330",
    "description": "2.5YR 2.5/2 - Reddish Black. Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable reddish-yellow influence from finely disseminated iron oxides.",
    "properties": ["Dominant Organic Matter", "Noticeable Iron Oxide Influence (Reddish-Yellow)", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "2.5YR 2.5/3": {
    "name": "Reddish Black",
    "hex": "#3E2F2B",
    "description": "2.5YR 2.5/3 - Reddish Black. Extremely low value, low-moderate chroma. A very dark soil with a more apparent reddish-yellow character from iron oxides compared to 2.5/2, still organic-rich.",
    "properties": ["High Organic Matter", "Clearer Reddish-Yellow Tinge from Iron Oxides", "Moist to Wet Environment", "Often acidic"]
  },
  "2.5YR 2.5/4": {
    "name": "Dark Reddish Brown",
    "hex": "#402A24",
    "description": "2.5YR 2.5/4 - Dark Reddish Brown. Extremely low value, moderate chroma. The darkest expression of a distinct reddish-yellow brown, indicating significant iron oxide content even under very low value (dark) conditions, likely mixed with organic material.",
    "properties": ["Significant Iron Oxides in a Very Dark Matrix", "Organic Material Mixture", "Low Light Reflectance", "Specific formation conditions"]
  },
  "5YR 8/1": {
    "name": "White",
    "hex": "#EBE4E0",
    "description": "5YR 8/1 - White with a very faint yellowish-pink cast. Very high value, very low chroma. Often reflects light-colored, sandy, or calcareous parent material with minimal iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated (Sandy/Calcareous)", "Minimal Iron Staining", "High Reflectance"]
  },
  "5YR 8/2": {
    "name": "Pinkish White",
    "hex": "#E7D8D2",
    "description": "5YR 8/2 - Pinkish White. Very high value, low chroma. A very light color with a noticeable warm pinkish-yellow (beige) tint from disseminated iron oxides.",
    "properties": ["Low Organic Matter", "Disseminated Iron Oxides", "Well-Aerated", "Light Mineral Base"]
  },
  "5YR 8/3": {
    "name": "Pink",
    "hex": "#E4CCC6",
    "description": "5YR 8/3 - Pink (warm, beige-pink). High value, low to moderate chroma. A pale but definite warm pink, indicating a greater presence of finely distributed iron oxides in a light matrix.",
    "properties": ["Finely Distributed Iron Oxides", "Good Aeration", "Low Organic Matter", "Often in well-drained sandy or loamy soils"]
  },
  "5YR 8/4": {
    "name": "Pink",
    "hex": "#E1C0BA",
    "description": "5YR 8/4 - Pink (warm). High value, moderate chroma. A clearer warm pink color, showing a more significant concentration of iron oxides within the light-colored soil material.",
    "properties": ["Moderate Iron Oxide Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "5YR 7/1": {
    "name": "Light Gray",
    "hex": "#D2CBC8",
    "description": "5YR 7/1 - Light Gray with a yellowish-pink hue. High value, very low chroma. Predominantly gray but with a discernible warm pinkish-yellow tint from slight iron oxide influence.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Iron Oxide Influence (Yellowish-Pink)", "Parent Material Color often visible"]
  },
  "5YR 7/2": {
    "name": "Pinkish Gray",
    "hex": "#CBBAB5",
    "description": "5YR 7/2 - Pinkish Gray (warm). High value, low chroma. A muted yellowish-red gray, indicating a mixture of light-colored minerals with some finely divided iron oxides.",
    "properties": ["Mixed Mineralogy", "Some Iron Oxides", "Low Organic Matter", "Generally Well-Drained"]
  },
  "5YR 7/3": {
    "name": "Pink",
    "hex": "#C5ACA7",
    "description": "5YR 7/3 - Pink (pale reddish-yellow). High value, low-moderate chroma. A pale warm pink or light beige-brown, characteristic of soils with a modest amount of mixed iron oxides.",
    "properties": ["Modest Iron Oxide Content", "Good Aeration", "Low Organic Matter"]
  },
  "5YR 7/4": {
    "name": "Pink",
    "hex": "#C09F99",
    "description": "5YR 7/4 - Pink (light reddish-yellow). High value, moderate chroma. A distinct light warm pink/light brown, showing a clear presence of iron oxides in a well-aerated environment.",
    "properties": ["Clear Iron Oxide Presence", "Well-Drained", "Oxidizing Environment"]
  },
  "5YR 7/6": {
    "name": "Reddish Yellow",
    "hex": "#B98B7F",
    "description": "5YR 7/6 - Reddish Yellow. High value, moderate to high chroma. A brighter, light reddish-yellow indicating a more substantial accumulation of mixed iron oxides (goethite and hematite).",
    "properties": ["Substantial Iron Oxides", "Good Aeration and Drainage", "Often from Weathering of various parent materials"]
  },
  "5YR 7/8": {
    "name": "Reddish Yellow",
    "hex": "#B27769",
    "description": "5YR 7/8 - Reddish Yellow. High value, high chroma. A strong and clear light reddish-yellow, signifying abundant and relatively pure mixed iron oxides.",
    "properties": ["Abundant Iron Oxides", "Well-Oxidized", "May indicate highly weathered conditions or specific parent rocks with iron accumulation"]
  },
  "5YR 6/1": {
    "name": "Gray",
    "hex": "#B3AEAA",
    "description": "5YR 6/1 - Gray with a yellowish-pink hue. Moderate value, very low chroma. A neutral gray appearance with a subtle warm influence.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Yellowish-Pink Tinge (Iron Oxides)"]
  },
  "5YR 6/2": {
    "name": "Pinkish Gray",
    "hex": "#AB9F9A",
    "description": "5YR 6/2 - Pinkish Gray (warm tone). Moderate value, low chroma. A muted reddish-yellow color mixed with gray, indicating some iron oxidation but not dominant.",
    "properties": ["Some Oxidized Iron", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "5YR 6/3": {
    "name": "Light Reddish Brown",
    "hex": "#A49089",
    "description": "5YR 6/3 - Light Reddish Brown. Moderate value, low-moderate chroma. A common color for less weathered soils or those with moderate iron content, often a pale brown.",
    "properties": ["Mixed Iron Oxides and Light Minerals", "Fair Drainage", "May have some organic influence dulling the color"]
  },
  "5YR 6/4": {
    "name": "Light Reddish Brown",
    "hex": "#9D8078",
    "description": "5YR 6/4 - Light Reddish Brown. Moderate value, moderate chroma. A more distinct but still light reddish-brown, indicating a good presence of iron oxides.",
    "properties": ["Good Iron Oxide Presence", "Well-Aerated", "Moderate Drainage"]
  },
  "5YR 6/6": {
    "name": "Reddish Yellow",
    "hex": "#956C60",
    "description": "5YR 6/6 - Reddish Yellow. Moderate value, moderate to high chroma. A clear reddish-yellow, typical of soils with significant mixed iron oxide accumulation, often in subsoils.",
    "properties": ["Significant Iron Oxide Accumulation", "Well-Drained", "Often in B horizons (cambic, argillic) with good aeration"]
  },
  "5YR 6/8": {
    "name": "Yellowish Red",
    "hex": "#8E584B",
    "description": "5YR 6/8 - Yellowish Red. Moderate value, high chroma. A strong yellowish-red, indicating a high concentration of mixed iron oxides (hematite becoming more influential), common in weathered soils.",
    "properties": ["High Iron Oxide Concentration", "Intense Weathering Product", "Well-Oxidized Environment"]
  },
  "5YR 5/1": {
    "name": "Gray",
    "hex": "#817E7A",
    "description": "5YR 5/1 - Gray. Moderate to low value, very low chroma. A neutral gray with potentially a very faint warm hue, often indicating parent material color or slight gleying if mottled.",
    "properties": ["Parent Material Influence", "Low Organic Matter or Gleyed", "Variable Drainage Conditions"]
  },
  "5YR 5/2": {
    "name": "Grayish Brown",
    "hex": "#7D716D",
    "description": "5YR 5/2 - Grayish Brown. Moderate to low value, low chroma. A dull, weak brown, suggesting limited iron oxide expression or mixing with organic matter and other minerals.",
    "properties": ["Limited Iron Oxide Expression", "Organic Matter Mixing Possible", "Moderate Aeration"]
  },
  "5YR 5/3": {
    "name": "Brown",
    "hex": "#7B6660",
    "description": "5YR 5/3 - Brown. Moderate to low value, low-moderate chroma. A common soil brown, often found in A or B horizons with moderate iron oxides and some organic incorporation.",
    "properties": ["Moderate Iron Oxides", "Some Organic Incorporation", "Generally Good Soil Structure", "Versatile Soil"]
  },
  "5YR 5/4": {
    "name": "Reddish Brown",
    "hex": "#795D56",
    "description": "5YR 5/4 - Reddish Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good iron oxide presence, often fertile.",
    "properties": ["Well-Aerated", "Good Iron Oxide Presence", "Often Good Agricultural Soils", "Balanced Properties"]
  },
  "5YR 5/6": {
    "name": "Yellowish Red",
    "hex": "#764F46",
    "description": "5YR 5/6 - Yellowish Red. Moderate to low value, moderate to high chroma. A clear, strong yellowish-red, indicating high amounts of mixed iron oxides and good oxidation.",
    "properties": ["High Iron Oxide Content", "Good Drainage and Aeration", "Often found in well-developed, weathered soil profiles"]
  },
  "5YR 5/8": {
    "name": "Yellowish Red",
    "hex": "#733F35",
    "description": "5YR 5/8 - Yellowish Red. Moderate to low value, high chroma. A very strong, relatively dark yellowish-red. Indicates a very high concentration of iron oxides and strong weathering.",
    "properties": ["Very High Iron Oxide Content", "Intensely Oxidized", "Common in tropical or subtropical regions (e.g., Oxisols, Ultisols)"]
  },
  "5YR 4/1": {
    "name": "Dark Gray",
    "hex": "#615E5C",
    "description": "5YR 4/1 - Dark Gray. Low value, very low chroma. Dark color, often due to significant organic matter content, with a very faint warm hue from underlying iron oxides.",
    "properties": ["Significant Organic Matter Influence", "Very Subtle Iron Oxide Presence", "Moisture Retentive", "Often fertile A-horizons"]
  },
  "5YR 4/2": {
    "name": "Dark Grayish Brown",
    "hex": "#5F5450",
    "description": "5YR 4/2 - Dark Grayish Brown. Low value, low chroma. A dark soil with a muted brown color, suggesting a mix of considerable organic matter and some iron oxides.",
    "properties": ["Considerable Organic Matter", "Some Iron Oxides", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "5YR 4/3": {
    "name": "Dark Brown",
    "hex": "#5F4D47",
    "description": "5YR 4/3 - Dark Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides both contribute significantly to the color. A rich brown.",
    "properties": ["Good Organic Content", "Moderate Iron Oxides", "Good Soil Structure", "Productive for Agriculture"]
  },
  "5YR 4/4": {
    "name": "Dark Reddish Brown",
    "hex": "#60463E",
    "description": "5YR 4/4 - Dark Reddish Brown. Low value, moderate chroma. A rich, dark reddish-brown indicating good aeration, significant iron oxides, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Iron Content"]
  },
  "5YR 4/6": {
    "name": "Yellowish Red",
    "hex": "#5F3B30",
    "description": "5YR 4/6 - Yellowish Red. Low value, moderate to high chroma. A strong dark yellowish-red, suggesting high iron oxide content, possibly with some organic matter.",
    "properties": ["High Iron Oxide Content", "May have Some Organic Matter", "Good Drainage", "Often in stable, well-weathered landscapes"]
  },
  "5YR 3/1": {
    "name": "Very Dark Gray",
    "hex": "#4B4947",
    "description": "5YR 3/1 - Very Dark Gray. Very low value, very low chroma. Very dark, typically organic-rich soil with only a subtle hint of underlying warm minerals.",
    "properties": ["High Organic Matter Content", "Slight Warm Mineral Influence", "Often in moist to wet conditions", "Fertile when drained"]
  },
  "5YR 3/2": {
    "name": "Dark Grayish Brown",
    "hex": "#4A413D",
    "description": "5YR 3/2 - Dark Grayish Brown. Very low value, low chroma. A very dark, muted brown. High organic matter content strongly influences this color, mixed with some iron oxides.",
    "properties": ["Very High Organic Matter", "Iron Oxides Present but Masked", "Moisture Retentive", "Typically Fertile"]
  },
  "5YR 3/3": {
    "name": "Dark Brown",
    "hex": "#4C3D38",
    "description": "5YR 3/3 - Dark Brown. Very low value, low-moderate chroma. Dark brown color, suggesting a balance of significant organic matter and notable iron oxide content.",
    "properties": ["Significant Organic Matter and Iron Oxides", "Good Fertility", "Moist Environment common"]
  },
  "5YR 3/4": {
    "name": "Dark Reddish Brown",
    "hex": "#4E3730",
    "description": "5YR 3/4 - Dark Reddish Brown. Very low value, moderate chroma. A darker reddish-brown, where iron oxides are more expressed than in lower chromas but still within a dark matrix, likely organic-influenced.",
    "properties": ["Strong Iron Oxide Presence in Dark Matrix", "Moderate Organic Matter", "Good Aeration for a dark soil"]
  },
  "5YR 2.5/1": {
    "name": "Black",
    "hex": "#3C3B3A",
    "description": "5YR 2.5/1 - Black. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or material like charcoal, with the faintest hint of warmth.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained"]
  },
  "5YR 2.5/2": {
    "name": "Reddish Black",
    "hex": "#3B3330",
    "description": "5YR 2.5/2 - Reddish Black. Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable warm (yellowish-red) influence from finely disseminated iron oxides.",
    "properties": ["Dominant Organic Matter", "Noticeable Iron Oxide Influence (Yellowish-Red)", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "7.5YR 8/1": {
    "name": "White",
    "hex": "#EBE4E1",
    "description": "7.5YR 8/1 - White with a very faint warm pinkish-beige cast. Very high value, very low chroma. Often reflects light-colored parent material, like loess or alluvial deposits, with minimal iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated", "Minimal Iron Staining", "High Reflectance", "Possibly Calcareous"]
  },
  "7.5YR 8/2": {
    "name": "Pinkish White",
    "hex": "#E7D9D3",
    "description": "7.5YR 8/2 - Pinkish White. Very high value, low chroma. A very light color with a noticeable warm pinkish-beige tint from disseminated iron oxides.",
    "properties": ["Low Organic Matter", "Disseminated Iron Oxides", "Well-Aerated", "Light Mineral Base"]
  },
  "7.5YR 8/3": {
    "name": "Pink",
    "hex": "#E4CEC7",
    "description": "7.5YR 8/3 - Pink (warm beige-pink). High value, low to moderate chroma. A pale but definite warm pinkish-beige, indicating a greater presence of finely distributed iron oxides in a light matrix.",
    "properties": ["Finely Distributed Iron Oxides", "Good Aeration", "Low Organic Matter", "Often in well-drained sandy or loamy soils"]
  },
  "7.5YR 8/4": {
    "name": "Pink",
    "hex": "#E1C2BC",
    "description": "7.5YR 8/4 - Pink (warm). High value, moderate chroma. A clearer warm pinkish-beige color, showing a more significant concentration of iron oxides within the light-colored soil material.",
    "properties": ["Moderate Iron Oxide Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "7.5YR 7/1": {
    "name": "Light Gray",
    "hex": "#D2CAC9",
    "description": "7.5YR 7/1 - Light Gray with a warm pinkish-beige hue. High value, very low chroma. Predominantly gray but with a discernible warm tint from slight iron oxide influence.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Iron Oxide Influence (Warm Tint)", "Parent Material Color often visible"]
  },
  "7.5YR 7/2": {
    "name": "Pinkish Gray",
    "hex": "#CBBCB7",
    "description": "7.5YR 7/2 - Pinkish Gray (warm). High value, low chroma. A muted warm gray with a pinkish-beige influence, indicating a mixture of light-colored minerals with some finely divided iron oxides.",
    "properties": ["Mixed Mineralogy", "Some Iron Oxides", "Low Organic Matter", "Generally Well-Drained"]
  },
  "7.5YR 7/3": {
    "name": "Pink",
    "hex": "#C5AEA9",
    "description": "7.5YR 7/3 - Pink (pale warm beige). High value, low-moderate chroma. A pale warm pink or light beige, characteristic of soils with a modest amount of mixed iron oxides.",
    "properties": ["Modest Iron Oxide Content", "Good Aeration", "Low Organic Matter"]
  },
  "7.5YR 7/4": {
    "name": "Pink",
    "hex": "#C0A099",
    "description": "7.5YR 7/4 - Pink (light warm brown). High value, moderate chroma. A distinct light warm pink/light brown, showing a clear presence of iron oxides in a well-aerated environment.",
    "properties": ["Clear Iron Oxide Presence", "Well-Drained", "Oxidizing Environment"]
  },
  "7.5YR 7/6": {
    "name": "Reddish Yellow",
    "hex": "#B98C7F",
    "description": "7.5YR 7/6 - Reddish Yellow. High value, moderate to high chroma. A brighter, light reddish-yellow or light strong brown, indicating a more substantial accumulation of mixed iron oxides.",
    "properties": ["Substantial Iron Oxides", "Good Aeration and Drainage", "Often from Weathering of various parent materials"]
  },
  "7.5YR 7/8": {
    "name": "Reddish Yellow",
    "hex": "#B27868",
    "description": "7.5YR 7/8 - Reddish Yellow. High value, high chroma. A strong and clear light reddish-yellow or strong brown, signifying abundant and relatively pure mixed iron oxides.",
    "properties": ["Abundant Iron Oxides", "Well-Oxidized", "May indicate highly weathered conditions or specific parent rocks with iron accumulation"]
  },
  "7.5YR 6/1": {
    "name": "Gray",
    "hex": "#B3AEAA",
    "description": "7.5YR 6/1 - Gray with a warm beige hue. Moderate value, very low chroma. A neutral gray appearance with a subtle warm influence.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Warm Beige Tinge (Iron Oxides)"]
  },
  "7.5YR 6/2": {
    "name": "Light Brownish Gray",
    "hex": "#ABA09B",
    "description": "7.5YR 6/2 - Light Brownish Gray. Moderate value, low chroma. A muted brown color mixed with gray, indicating some iron oxidation but not dominant.",
    "properties": ["Some Oxidized Iron", "Mixed Mineral Soil", "Moderate Drainage"]
  },
  "7.5YR 6/3": {
    "name": "Light Brown",
    "hex": "#A49189",
    "description": "7.5YR 6/3 - Light Brown. Moderate value, low-moderate chroma. A common color for less weathered soils or those with moderate iron content, often a pale brown.",
    "properties": ["Mixed Iron Oxides and Light Minerals", "Fair Drainage", "May have some organic influence dulling the color"]
  },
  "7.5YR 6/4": {
    "name": "Light Brown",
    "hex": "#9D8178",
    "description": "7.5YR 6/4 - Light Brown. Moderate value, moderate chroma. A more distinct but still light brown, indicating a good presence of iron oxides.",
    "properties": ["Good Iron Oxide Presence", "Well-Aerated", "Moderate Drainage"]
  },
  "7.5YR 6/6": {
    "name": "Reddish Yellow",
    "hex": "#956D5F",
    "description": "7.5YR 6/6 - Reddish Yellow (Strong Brown). Moderate value, moderate to high chroma. A clear reddish-yellow or strong brown, typical of soils with significant mixed iron oxide accumulation, often in subsoils.",
    "properties": ["Significant Iron Oxide Accumulation", "Well-Drained", "Often in B horizons (cambic, argillic) with good aeration"]
  },
  "7.5YR 6/8": {
    "name": "Strong Brown",
    "hex": "#8E594A",
    "description": "7.5YR 6/8 - Strong Brown. Moderate value, high chroma. A strong brown with a reddish-yellow hue, indicating a high concentration of mixed iron oxides, common in weathered soils.",
    "properties": ["High Iron Oxide Concentration", "Intense Weathering Product", "Well-Oxidized Environment"]
  },
  "7.5YR 5/1": {
    "name": "Gray",
    "hex": "#817E7A",
    "description": "7.5YR 5/1 - Gray. Moderate to low value, very low chroma. A neutral gray with potentially a very faint warm hue, often indicating parent material color or slight gleying if mottled.",
    "properties": ["Parent Material Influence", "Low Organic Matter or Gleyed", "Variable Drainage Conditions"]
  },
  "7.5YR 5/2": {
    "name": "Brownish Gray",
    "hex": "#7D716D",
    "description": "7.5YR 5/2 - Brownish Gray. Moderate to low value, low chroma. A dull, weak brown, suggesting limited iron oxide expression or mixing with organic matter and other minerals.",
    "properties": ["Limited Iron Oxide Expression", "Organic Matter Mixing Possible", "Moderate Aeration"]
  },
  "7.5YR 5/3": {
    "name": "Brown",
    "hex": "#7B6660",
    "description": "7.5YR 5/3 - Brown. Moderate to low value, low-moderate chroma. A common soil brown, often found in A or B horizons with moderate iron oxides and some organic incorporation.",
    "properties": ["Moderate Iron Oxides", "Some Organic Incorporation", "Generally Good Soil Structure", "Versatile Soil"]
  },
  "7.5YR 5/4": {
    "name": "Brown",
    "hex": "#795D56",
    "description": "7.5YR 5/4 - Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good iron oxide presence, often fertile.",
    "properties": ["Well-Aerated", "Good Iron Oxide Presence", "Often Good Agricultural Soils", "Balanced Properties"]
  },
  "7.5YR 5/6": {
    "name": "Strong Brown",
    "hex": "#764F45",
    "description": "7.5YR 5/6 - Strong Brown. Moderate to low value, moderate to high chroma. A clear, strong brown with a reddish-yellow hue, indicating high amounts of mixed iron oxides and good oxidation.",
    "properties": ["High Iron Oxide Content", "Good Drainage and Aeration", "Often found in well-developed, weathered soil profiles"]
  },
  "7.5YR 5/8": {
    "name": "Strong Brown",
    "hex": "#733F34",
    "description": "7.5YR 5/8 - Strong Brown. Moderate to low value, high chroma. A very strong brown with a distinct reddish-yellow hue. Indicates a very high concentration of iron oxides and strong weathering.",
    "properties": ["Very High Iron Oxide Content", "Intensely Oxidized", "Common in tropical or subtropical regions (e.g., Oxisols, Ultisols)"]
  },
  "7.5YR 4/1": {
    "name": "Dark Gray",
    "hex": "#615E5C",
    "description": "7.5YR 4/1 - Dark Gray. Low value, very low chroma. Dark color, often due to significant organic matter content, with a very faint warm hue from underlying iron oxides.",
    "properties": ["Significant Organic Matter Influence", "Very Subtle Iron Oxide Presence (Warm Tint)", "Moisture Retentive", "Often fertile A-horizons"]
  },
  "7.5YR 4/2": {
    "name": "Dark Brownish Gray",
    "hex": "#5F5450",
    "description": "7.5YR 4/2 - Dark Brownish Gray. Low value, low chroma. A dark soil with a muted brown color, suggesting a mix of considerable organic matter and some iron oxides.",
    "properties": ["Considerable Organic Matter", "Some Iron Oxides", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "7.5YR 4/3": {
    "name": "Dark Brown",
    "hex": "#5F4D47",
    "description": "7.5YR 4/3 - Dark Brown. Low value, low-moderate chroma. Common in fertile topsoils where organic matter and iron oxides both contribute significantly to the color. A rich brown.",
    "properties": ["Good Organic Content", "Moderate Iron Oxides", "Good Soil Structure", "Productive for Agriculture"]
  },
  "7.5YR 4/4": {
    "name": "Dark Brown",
    "hex": "#60463E",
    "description": "7.5YR 4/4 - Dark Brown. Low value, moderate chroma. A rich, dark brown indicating good aeration, significant iron oxides, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Iron Content"]
  },
  "7.5YR 4/6": {
    "name": "Strong Brown",
    "hex": "#5F3B30",
    "description": "7.5YR 4/6 - Strong Brown. Low value, moderate to high chroma. A strong dark brown with a reddish-yellow hue, suggesting high iron oxide content, possibly with some organic matter.",
    "properties": ["High Iron Oxide Content", "May have Some Organic Matter", "Good Drainage", "Often in stable, well-weathered landscapes"]
  },
  "7.5YR 3/1": {
    "name": "Very Dark Gray",
    "hex": "#4B4947",
    "description": "7.5YR 3/1 - Very Dark Gray. Very low value, very low chroma. Very dark, typically organic-rich soil with only a subtle hint of underlying warm minerals.",
    "properties": ["High Organic Matter Content", "Slight Warm Mineral Influence", "Often in moist to wet conditions", "Fertile when drained"]
  },
  "7.5YR 3/2": {
    "name": "Dark Brown",
    "hex": "#4A413D",
    "description": "7.5YR 3/2 - Dark Brown. Very low value, low chroma. A very dark, muted brown. High organic matter content strongly influences this color, mixed with some iron oxides.",
    "properties": ["Very High Organic Matter", "Iron Oxides Present but Masked", "Moisture Retentive", "Typically Fertile"]
  },
  "7.5YR 3/3": {
    "name": "Dark Brown",
    "hex": "#4C3D38",
    "description": "7.5YR 3/3 - Dark Brown. Very low value, low-moderate chroma. Dark brown color, suggesting a balance of significant organic matter and notable iron oxide content.",
    "properties": ["Significant Organic Matter and Iron Oxides", "Good Fertility", "Moist Environment common"]
  },
  "7.5YR 3/4": {
    "name": "Dark Brown",
    "hex": "#4E3730",
    "description": "7.5YR 3/4 - Dark Brown. Very low value, moderate chroma. A darker brown with a more expressed reddish-yellow hue from iron oxides, still within a dark matrix, likely organic-influenced.",
    "properties": ["Strong Iron Oxide Presence in Dark Matrix", "Moderate Organic Matter", "Good Aeration for a dark soil"]
  },
  "7.5YR 2.5/1": {
    "name": "Black",
    "hex": "#3C3B3A",
    "description": "7.5YR 2.5/1 - Black. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck) or material like charcoal, with the faintest hint of warmth.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained"]
  },
  "7.5YR 2.5/2": {
    "name": "Black",
    "hex": "#3B3330",
    "description": "7.5YR 2.5/2 - Black (with brownish cast). Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable warm (brownish) influence from finely disseminated iron oxides.",
    "properties": ["Dominant Organic Matter", "Noticeable Iron Oxide Influence (Brownish)", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "7.5YR 2.5/3": {
    "name": "Very Dark Brown",
    "hex": "#3E2F2B",
    "description": "7.5YR 2.5/3 - Very Dark Brown. Extremely low value, low-moderate chroma. A very dark soil with a more apparent warm brown character from iron oxides compared to 2.5/2, still organic-rich.",
    "properties": ["High Organic Matter", "Clearer Warm Brown Tinge from Iron Oxides", "Moist to Wet Environment", "Often acidic"]
  },
  "10YR 8/1": {
    "name": "White",
    "hex": "#EDEAE1",
    "description": "10YR 8/1 - White with a very faint yellowish cast. Very high value, very low chroma. Often indicates parent material like chalk, marl, or highly leached sands with minimal organic matter or iron staining.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated (Calcareous/Sandy)", "Minimal Iron Staining", "High Reflectance"]
  },
  "10YR 8/2": {
    "name": "Very Pale Brown",
    "hex": "#E7D9CD",
    "description": "10YR 8/2 - Very Pale Brown. Very high value, low chroma. A very light, off-white beige or pale yellowish-brown, typical of arid soils or C horizons low in organic matter.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Arid/Semi-Arid Conditions Possible", "Well-Aerated if not cemented"]
  },
  "10YR 8/3": {
    "name": "Very Pale Brown",
    "hex": "#E4CEC1",
    "description": "10YR 8/3 - Very Pale Brown. High value, low to moderate chroma. A light yellowish-beige, indicating some iron oxides (goethite) in a light matrix.",
    "properties": ["Some Goethite Presence", "Good Aeration", "Low Organic Matter", "Often in well-drained loamy or sandy soils"]
  },
  "10YR 8/4": {
    "name": "Very Pale Brown",
    "hex": "#E1C3B5",
    "description": "10YR 8/4 - Very Pale Brown. High value, moderate chroma. A clearer light yellowish-beige, showing a more significant concentration of goethite within the light-colored soil material.",
    "properties": ["Moderate Goethite Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "10YR 8/6": {
    "name": "Yellow",
    "hex": "#DCAE97",
    "description": "10YR 8/6 - Yellow. High value, moderate to high chroma. A light, clear yellow or yellowish-brown, indicating a higher concentration of yellowish iron oxides like goethite or limonite.",
    "properties": ["Higher Goethite/Limonite", "Well-Oxidized", "Good Drainage", "Low Organic Matter", "Uncommon in many temperate soils, more in specific parent materials"]
  },
  "10YR 8/8": {
    "name": "Yellow",
    "hex": "#D79A7A",
    "description": "10YR 8/8 - Yellow. High value, high chroma. A strong, bright light yellow or yellowish-orange. Suggests very high concentration of specific yellowish iron minerals.",
    "properties": ["Concentrated Yellow Iron Oxides", "Very Well-Aerated", "Specific Geological Context Often", "Rare for typical soils"]
  },
  "10YR 7/1": {
    "name": "Light Gray",
    "hex": "#D2CAC0",
    "description": "10YR 7/1 - Light Gray. High value, very low chroma. Predominantly gray with a faint yellowish or brownish tint. Often indicates low organic matter and minimal iron oxide development or leaching.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Yellowish/Brownish Tint", "Parent Material Color or Leached Horizon (E horizon)"]
  },
  "10YR 7/2": {
    "name": "Light Gray",
    "hex": "#CBBBAF",
    "description": "10YR 7/2 - Light Gray. High value, low chroma. A light grayish-brown or pale beige, indicating some iron oxide presence but still very light.",
    "properties": ["Some Iron Oxides (Goethite)", "Low Organic Matter", "Generally Well-Drained"]
  },
  "10YR 7/3": {
    "name": "Very Pale Brown",
    "hex": "#C5AE9F",
    "description": "10YR 7/3 - Very Pale Brown. High value, low-moderate chroma. A pale yellowish-brown, characteristic of soils with modest amounts of goethite and low organic matter.",
    "properties": ["Modest Goethite Content", "Good Aeration", "Low Organic Matter"]
  },
  "10YR 7/4": {
    "name": "Very Pale Brown",
    "hex": "#C0A18F",
    "description": "10YR 7/4 - Very Pale Brown. High value, moderate chroma. A distinct light yellowish-brown, showing a clear presence of goethite in a well-aerated environment.",
    "properties": ["Clear Goethite Presence", "Well-Drained", "Oxidizing Environment", "Common in loess or alluvial soils"]
  },
  "10YR 7/6": {
    "name": "Yellow",
    "hex": "#B98D76",
    "description": "10YR 7/6 - Yellow. High value, moderate to high chroma. A brighter, yellowish-brown indicating a more substantial accumulation of goethite.",
    "properties": ["Substantial Goethite", "Good Aeration and Drainage", "Often from Weathering of various parent materials"]
  },
  "10YR 7/8": {
    "name": "Yellow",
    "hex": "#B27959",
    "description": "10YR 7/8 - Yellow. High value, high chroma. A strong and clear yellowish-orange brown, signifying abundant and relatively pure goethite or similar yellowish iron oxides.",
    "properties": ["Abundant Yellow Iron Oxides", "Well-Oxidized", "May indicate specific parent materials or conditions favoring goethite formation"]
  },
  "10YR 6/1": {
    "name": "Gray",
    "hex": "#B3AEA1",
    "description": "10YR 6/1 - Gray. Moderate value, very low chroma. A neutral gray appearance with a very subtle yellowish or brownish influence, often from parent material or slight gleying if mottles are present.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Yellowish/Brownish Tinge", "Possible Gleying if mottled"]
  },
  "10YR 6/2": {
    "name": "Light Brownish Gray",
    "hex": "#ABA092",
    "description": "10YR 6/2 - Light Brownish Gray. Moderate value, low chroma. A muted brownish-gray, indicating some iron oxidation (goethite) but not dominant over the base mineral color or some organic dulling.",
    "properties": ["Some Oxidized Iron (Goethite)", "Mixed Mineral Soil", "Moderate Drainage", "Low to Moderate Organic Matter"]
  },
  "10YR 6/3": {
    "name": "Pale Brown",
    "hex": "#A49182",
    "description": "10YR 6/3 - Pale Brown. Moderate value, low-moderate chroma. A lighter, less saturated brown, often where goethite is present but mixed with other lighter minerals or slightly masked.",
    "properties": ["Mixed Goethite and Light Minerals", "Fair Drainage", "Common in C horizons or less developed soils"]
  },
  "10YR 6/4": {
    "name": "Light Yellowish Brown",
    "hex": "#9D8171",
    "description": "10YR 6/4 - Light Yellowish Brown. Moderate value, moderate chroma. A more distinct but still light yellowish-brown, indicating a good presence of goethite.",
    "properties": ["Good Goethite Presence", "Well-Aerated", "Moderate Drainage", "Often in Bw or BC horizons"]
  },
  "10YR 6/6": {
    "name": "Brownish Yellow",
    "hex": "#956E58",
    "description": "10YR 6/6 - Brownish Yellow. Moderate value, moderate to high chroma. A clear brownish-yellow or light yellowish-brown, typical of soils with significant goethite accumulation.",
    "properties": ["Significant Goethite Accumulation", "Well-Drained", "Often in cambic horizons or well-oxidized parent materials"]
  },
  "10YR 6/8": {
    "name": "Brownish Yellow",
    "hex": "#8E5A3F",
    "description": "10YR 6/8 - Brownish Yellow. Moderate value, high chroma. A strong brownish-yellow, indicating a high concentration of goethite, common in well-weathered soils.",
    "properties": ["High Goethite Concentration", "Intense Weathering Product (for goethite)", "Well-Oxidized Environment"]
  },
  "10YR 5/1": {
    "name": "Gray",
    "hex": "#817E71",
    "description": "10YR 5/1 - Gray. Moderate to low value, very low chroma. A darker gray with a faint brownish or yellowish hue. Can indicate some organic matter accumulation or gleying conditions if mottles are present.",
    "properties": ["Some Organic Matter or Gleyed", "Faint Brownish/Yellowish Hue", "Variable Drainage Conditions"]
  },
  "10YR 5/2": {
    "name": "Grayish Brown",
    "hex": "#7D7164",
    "description": "10YR 5/2 - Grayish Brown. Moderate to low value, low chroma. A dull, weak brown, suggesting limited iron oxide expression or significant mixing with organic matter and other minerals.",
    "properties": ["Limited Iron Oxide Expression", "Organic Matter Mixing Common", "Moderate Aeration", "Often in A horizons"]
  },
  "10YR 5/3": {
    "name": "Brown",
    "hex": "#7B6659",
    "description": "10YR 5/3 - Brown. Moderate to low value, low-moderate chroma. A very common soil brown, typical of A or B horizons with moderate goethite and some organic incorporation.",
    "properties": ["Moderate Goethite", "Some Organic Incorporation", "Generally Good Soil Structure", "Versatile Agricultural Soil"]
  },
  "10YR 5/4": {
    "name": "Yellowish Brown",
    "hex": "#795D4F",
    "description": "10YR 5/4 - Yellowish Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good goethite presence, often fertile.",
    "properties": ["Well-Aerated", "Good Goethite Presence", "Often Good Agricultural Soils", "Balanced Properties"]
  },
  "10YR 5/6": {
    "name": "Yellowish Brown",
    "hex": "#764F3E",
    "description": "10YR 5/6 - Yellowish Brown. Moderate to low value, moderate to high chroma. A clear, strong yellowish-brown, indicating high amounts of goethite and good oxidation.",
    "properties": ["High Goethite Content", "Good Drainage and Aeration", "Often found in well-developed B horizons (cambic, argillic)"]
  },
  "10YR 5/8": {
    "name": "Yellowish Brown",
    "hex": "#733F2A",
    "description": "10YR 5/8 - Yellowish Brown. Moderate to low value, high chroma. A very strong, relatively dark yellowish-brown. Indicates a very high concentration of goethite and strong weathering.",
    "properties": ["Very High Goethite Content", "Intensely Oxidized", "Common in well-weathered profiles in temperate to subtropical regions"]
  },
  "10YR 4/1": {
    "name": "Dark Gray",
    "hex": "#615E55",
    "description": "10YR 4/1 - Dark Gray. Low value, very low chroma. Dark color, often due to significant organic matter content, with a faint brownish or yellowish hue from underlying iron oxides.",
    "properties": ["Significant Organic Matter Influence", "Subtle Goethite/Iron Oxide Presence", "Moisture Retentive", "Fertile A-horizons (e.g., Mollic epipedon)"]
  },
  "10YR 4/2": {
    "name": "Dark Grayish Brown",
    "hex": "#5F544A",
    "description": "10YR 4/2 - Dark Grayish Brown. Low value, low chroma. A dark soil with a muted brown color, suggesting a mix of considerable organic matter and some goethite.",
    "properties": ["Considerable Organic Matter", "Some Goethite", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "10YR 4/3": {
    "name": "Dark Brown",
    "hex": "#5F4D41",
    "description": "10YR 4/3 - Dark Brown. Low value, low-moderate chroma. Common in fertile topsoils (A horizons) where organic matter and goethite both contribute significantly to the color. A rich brown.",
    "properties": ["Good Organic Content", "Moderate Goethite", "Excellent Soil Structure", "Highly Productive for Agriculture"]
  },
  "10YR 4/4": {
    "name": "Dark Yellowish Brown",
    "hex": "#604637",
    "description": "10YR 4/4 - Dark Yellowish Brown. Low value, moderate chroma. A rich, dark yellowish-brown indicating good aeration, significant goethite, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Goethite Content"]
  },
  "10YR 4/6": {
    "name": "Dark Yellowish Brown",
    "hex": "#5F3B29",
    "description": "10YR 4/6 - Dark Yellowish Brown. Low value, moderate to high chroma. A strong dark yellowish-brown, suggesting high goethite content, possibly with some organic matter.",
    "properties": ["High Goethite Content", "May have Some Organic Matter", "Good Drainage", "Often in B horizons of well-drained soils"]
  },
  "10YR 3/1": {
    "name": "Very Dark Gray",
    "hex": "#4B4941",
    "description": "10YR 3/1 - Very Dark Gray. Very low value, very low chroma. Very dark, typically organic-rich soil (e.g., A horizon of Mollisols) with only a subtle hint of underlying brownish minerals.",
    "properties": ["High Organic Matter Content (Humus)", "Slight Brownish Mineral Influence", "Often in moist conditions", "Very Fertile"]
  },
  "10YR 3/2": {
    "name": "Very Dark Grayish Brown",
    "hex": "#4A4137",
    "description": "10YR 3/2 - Very Dark Grayish Brown. Very low value, low chroma. A very dark, muted brown. High organic matter content strongly influences this color, mixed with some goethite.",
    "properties": ["Very High Organic Matter", "Goethite Present but Masked", "Moisture Retentive", "Highly Fertile (e.g., mollic or umbric epipedon)"]
  },
  "10YR 3/3": {
    "name": "Dark Brown",
    "hex": "#4C3D31",
    "description": "10YR 3/3 - Dark Brown. Very low value, low-moderate chroma. Dark brown color, suggesting a balance of significant organic matter and notable goethite content.",
    "properties": ["Significant Organic Matter and Goethite", "Excellent Fertility", "Moist Environment common", "Good for diverse agriculture"]
  },
  "10YR 3/4": {
    "name": "Dark Yellowish Brown",
    "hex": "#4E3729",
    "description": "10YR 3/4 - Dark Yellowish Brown. Very low value, moderate chroma. A darker yellowish-brown, where goethite is more expressed than in lower chromas but still within a dark, organic-rich matrix.",
    "properties": ["Strong Goethite Presence in Dark Matrix", "Moderate to High Organic Matter", "Well-Aerated for a dark soil"]
  },
  "10YR 2/1": {
    "name": "Black",
    "hex": "#3C3B35",
    "description": "10YR 2/1 - Black. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck or peat) or material like charcoal, with the faintest hint of brown.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained", "Acidic if peat"]
  },
  "10YR 2/2": {
    "name": "Very Dark Brown",
    "hex": "#3B332A",
    "description": "10YR 2/2 - Very Dark Brown (nearly black). Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable brownish influence from finely disseminated goethite.",
    "properties": ["Dominant Organic Matter", "Noticeable Goethite Influence (Brownish)", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "2.5Y 8/1": {
    "name": "White",
    "hex": "#EBE5E0",
    "description": "2.5Y 8/1 - White with a very faint yellow cast. Very high value, very low chroma. Often indicates light-colored parent material such as chalk, marl, or highly bleached sand with minimal iron or organic influence.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated (Calcareous/Sandy)", "Minimal Iron Staining (Yellowish)", "High Reflectance"]
  },
  "2.5Y 8/2": {
    "name": "White",
    "hex": "#E7DACD",
    "description": "2.5Y 8/2 - White (with a pale yellow tint). Very high value, low chroma. A very light, off-white with a more noticeable pale yellow tint from disseminated yellowish iron oxides or other minerals.",
    "properties": ["Low Organic Matter", "Disseminated Yellowish Minerals/Iron Oxides", "Well-Aerated", "Light Mineral Base"]
  },
  "2.5Y 8/3": {
    "name": "Pale Yellow",
    "hex": "#E4D0C1",
    "description": "2.5Y 8/3 - Pale Yellow. High value, low to moderate chroma. A light, clear pale yellow, indicating some presence of yellowish iron compounds like goethite or jarosite in a light matrix.",
    "properties": ["Some Goethite/Jarosite Presence", "Good Aeration", "Low Organic Matter", "Often in well-drained loamy or sandy soils, or specific deposits"]
  },
  "2.5Y 8/4": {
    "name": "Pale Yellow",
    "hex": "#E1C6B5",
    "description": "2.5Y 8/4 - Pale Yellow. High value, moderate chroma. A clearer light yellow, showing a more significant concentration of yellowish iron oxides within the light-colored soil material.",
    "properties": ["Moderate Yellowish Iron Oxide Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "2.5Y 8/6": {
    "name": "Yellow",
    "hex": "#DCAF97",
    "description": "2.5Y 8/6 - Yellow. High value, moderate to high chroma. A light, distinct yellow, indicating a higher concentration of yellowish iron oxides like goethite, or potentially jarosite.",
    "properties": ["Higher Goethite/Jarosite", "Well-Oxidized", "Good Drainage", "Low Organic Matter", "Can indicate specific mineralogy"]
  },
  "2.5Y 8/8": {
    "name": "Yellow",
    "hex": "#D79B79",
    "description": "2.5Y 8/8 - Yellow. High value, high chroma. A strong, bright light yellow. Suggests very high concentration of specific yellowish iron minerals, less common in typical soils.",
    "properties": ["Concentrated Yellow Iron Oxides", "Very Well-Aerated", "Specific Geological Context Often", "Rare for most agricultural soils"]
  },
  "2.5Y 7/1": {
    "name": "Light Gray",
    "hex": "#D2CACC",
    "description": "2.5Y 7/1 - Light Gray with a faint yellow tint. High value, very low chroma. Predominantly gray with a barely discernible yellowish tint. Often indicates low organic matter and minimal iron oxide development or leaching.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Yellowish Tint", "Parent Material Color or Leached Horizon (E horizon)"]
  },
  "2.5Y 7/2": {
    "name": "Light Gray",
    "hex": "#CBBCB0",
    "description": "2.5Y 7/2 - Light Gray (with pale yellow cast). High value, low chroma. A light grayish-yellow or pale beige, indicating some yellowish iron oxide presence but still very light.",
    "properties": ["Some Yellowish Iron Oxides (Goethite)", "Low Organic Matter", "Generally Well-Drained"]
  },
  "2.5Y 7/3": {
    "name": "Pale Yellow",
    "hex": "#C5AF9F",
    "description": "2.5Y 7/3 - Pale Yellow. High value, low-moderate chroma. A pale yellowish-brown or light olive gray, characteristic of soils with modest amounts of goethite and low organic matter.",
    "properties": ["Modest Goethite Content", "Good Aeration", "Low Organic Matter", "Can be found in loess or alluvium"]
  },
  "2.5Y 7/4": {
    "name": "Pale Yellow",
    "hex": "#C0A28F",
    "description": "2.5Y 7/4 - Pale Yellow. High value, moderate chroma. A distinct light yellowish-brown, showing a clear presence of goethite in a well-aerated environment.",
    "properties": ["Clear Goethite Presence", "Well-Drained", "Oxidizing Environment", "Common in some sedimentary deposits"]
  },
  "2.5Y 7/6": {
    "name": "Yellow",
    "hex": "#B98E75",
    "description": "2.5Y 7/6 - Yellow. High value, moderate to high chroma. A brighter, yellowish-brown indicating a more substantial accumulation of goethite or other yellow iron pigments.",
    "properties": ["Substantial Goethite", "Good Aeration and Drainage", "Often from Weathering of various parent materials"]
  },
  "2.5Y 7/8": {
    "name": "Yellow",
    "hex": "#B27A58",
    "description": "2.5Y 7/8 - Yellow. High value, high chroma. A strong and clear yellow or yellowish-orange brown, signifying abundant and relatively pure goethite or similar yellowish iron oxides.",
    "properties": ["Abundant Yellow Iron Oxides", "Well-Oxidized", "May indicate specific parent materials or conditions favoring goethite/jarosite formation"]
  },
  "2.5Y 6/1": {
    "name": "Gray",
    "hex": "#B3AE9A",
    "description": "2.5Y 6/1 - Gray with a yellow tint. Moderate value, very low chroma. A neutral gray appearance with a subtle yellowish or olive influence, often from parent material or slight gleying.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Yellowish/Olive Tinge", "Possible Gleying if mottled"]
  },
  "2.5Y 6/2": {
    "name": "Light Grayish Brown",
    "hex": "#ABA08B",
    "description": "2.5Y 6/2 - Light Grayish Brown (olive cast). Moderate value, low chroma. A muted brownish-gray with a distinct yellowish or olive hue, indicating some goethite or other Fe-bearing minerals.",
    "properties": ["Some Oxidized Iron (Goethite/Olivine weathering)", "Mixed Mineral Soil", "Moderate Drainage", "Low to Moderate Organic Matter"]
  },
  "2.5Y 6/3": {
    "name": "Light Yellowish Brown",
    "hex": "#A4917B",
    "description": "2.5Y 6/3 - Light Yellowish Brown. Moderate value, low-moderate chroma. A lighter, less saturated yellowish-brown or light olive brown, where goethite is present.",
    "properties": ["Goethite Presence", "Fair Drainage", "Common in C horizons or less developed soils"]
  },
  "2.5Y 6/4": {
    "name": "Light Yellowish Brown",
    "hex": "#9D826A",
    "description": "2.5Y 6/4 - Light Yellowish Brown. Moderate value, moderate chroma. A more distinct light yellowish-brown or olive brown, indicating a good presence of goethite.",
    "properties": ["Good Goethite Presence", "Well-Aerated", "Moderate Drainage", "Often in Bw or BC horizons"]
  },
  "2.5Y 6/6": {
    "name": "Olive Yellow",
    "hex": "#956F50",
    "description": "2.5Y 6/6 - Olive Yellow. Moderate value, moderate to high chroma. A clear olive yellow or brownish-yellow, typical of soils with significant goethite accumulation.",
    "properties": ["Significant Goethite Accumulation", "Well-Drained", "Often in cambic horizons or well-oxidized parent materials"]
  },
  "2.5Y 6/8": {
    "name": "Olive Yellow",
    "hex": "#8E5B37",
    "description": "2.5Y 6/8 - Olive Yellow. Moderate value, high chroma. A strong olive yellow or brownish-yellow, indicating a high concentration of goethite, common in well-weathered soils.",
    "properties": ["High Goethite Concentration", "Intense Weathering Product (for goethite)", "Well-Oxidized Environment"]
  },
  "2.5Y 5/1": {
    "name": "Gray",
    "hex": "#817E6A",
    "description": "2.5Y 5/1 - Gray (olive gray). Moderate to low value, very low chroma. A darker gray with a distinct olive or yellowish hue. Can indicate some organic matter accumulation or gleying conditions.",
    "properties": ["Some Organic Matter or Gleyed", "Distinct Olive/Yellowish Hue", "Variable Drainage Conditions"]
  },
  "2.5Y 5/2": {
    "name": "Grayish Brown",
    "hex": "#7D715D",
    "description": "2.5Y 5/2 - Grayish Brown (olive cast). Moderate to low value, low chroma. A dull, weak olive brown, suggesting limited iron oxide expression or significant mixing with organic matter.",
    "properties": ["Limited Iron Oxide Expression", "Organic Matter Mixing Common", "Moderate Aeration", "Often in A horizons"]
  },
  "2.5Y 5/3": {
    "name": "Olive Brown",
    "hex": "#7B6652",
    "description": "2.5Y 5/3 - Olive Brown. Moderate to low value, low-moderate chroma. A common soil brown with a distinct olive/yellow hue, typical of A or B horizons with moderate goethite and some organic incorporation.",
    "properties": ["Moderate Goethite (Olive tone)", "Some Organic Incorporation", "Generally Good Soil Structure"]
  },
  "2.5Y 5/4": {
    "name": "Olive Brown",
    "hex": "#795D48",
    "description": "2.5Y 5/4 - Olive Brown. Moderate to low value, moderate chroma. A common soil color indicating well-aerated conditions with good goethite presence yielding an olive brown.",
    "properties": ["Well-Aerated", "Good Goethite Presence (Olive)", "Often Good Agricultural Soils"]
  },
  "2.5Y 5/6": {
    "name": "Yellowish Brown",
    "hex": "#764F36",
    "description": "2.5Y 5/6 - Yellowish Brown. Moderate to low value, moderate to high chroma. A clear, strong yellowish-brown or olive brown, indicating high amounts of goethite and good oxidation.",
    "properties": ["High Goethite Content", "Good Drainage and Aeration", "Often found in well-developed B horizons"]
  },
  "2.5Y 4/1": {
    "name": "Dark Gray",
    "hex": "#615E4E",
    "description": "2.5Y 4/1 - Dark Gray (dark olive gray). Low value, very low chroma. Dark color, often due to significant organic matter content, with a distinct olive or yellowish hue from underlying iron oxides or minerals.",
    "properties": ["Significant Organic Matter Influence", "Subtle Goethite/Olive Mineral Presence", "Moisture Retentive", "Fertile A-horizons"]
  },
  "2.5Y 4/2": {
    "name": "Dark Grayish Brown",
    "hex": "#5F5443",
    "description": "2.5Y 4/2 - Dark Grayish Brown (dark olive brown). Low value, low chroma. A dark soil with a muted olive brown color, suggesting a mix of considerable organic matter and some goethite/olivine weathering products.",
    "properties": ["Considerable Organic Matter", "Some Goethite/Olivine Influence", "Typically Fertile", "Good Moisture Holding Capacity"]
  },
  "2.5Y 4/3": {
    "name": "Dark Olive Brown",
    "hex": "#5F4D3A",
    "description": "2.5Y 4/3 - Dark Olive Brown. Low value, low-moderate chroma. Common in fertile topsoils (A horizons) where organic matter and goethite (giving olive tone) both contribute significantly to the color.",
    "properties": ["Good Organic Content", "Moderate Goethite (Olive)", "Excellent Soil Structure", "Highly Productive"]
  },
  "2.5Y 4/4": {
    "name": "Dark Yellowish Brown",
    "hex": "#604630",
    "description": "2.5Y 4/4 - Dark Yellowish Brown (dark olive brown). Low value, moderate chroma. A rich, dark olive brown or dark yellowish-brown indicating good aeration, significant goethite, and substantial organic matter.",
    "properties": ["Well-Aerated", "Productive Soil Horizon", "Balanced Organic and High Goethite Content"]
  },
  "2.5Y 3/1": {
    "name": "Very Dark Gray",
    "hex": "#4B493A",
    "description": "2.5Y 3/1 - Very Dark Gray (very dark olive gray). Very low value, very low chroma. Very dark, typically organic-rich soil with a subtle hint of underlying olive/yellowish minerals.",
    "properties": ["High Organic Matter Content (Humus)", "Slight Olive/Yellowish Mineral Influence", "Often in moist conditions", "Very Fertile"]
  },
  "2.5Y 3/2": {
    "name": "Very Dark Grayish Brown",
    "hex": "#4A4130",
    "description": "2.5Y 3/2 - Very Dark Grayish Brown (very dark olive brown). Very low value, low chroma. A very dark, muted olive brown. High organic matter content strongly influences this color, mixed with some goethite.",
    "properties": ["Very High Organic Matter", "Goethite Present but Masked (Olive tone)", "Moisture Retentive", "Highly Fertile"]
  },
  "2.5Y 3/3": {
    "name": "Dark Olive Brown",
    "hex": "#4C3D29",
    "description": "2.5Y 3/3 - Dark Olive Brown. Very low value, low-moderate chroma. Dark olive brown color, suggesting a balance of significant organic matter and notable goethite content yielding an olive hue.",
    "properties": ["Significant Organic Matter and Goethite (Olive)", "Excellent Fertility", "Moist Environment common", "Good for diverse agriculture"]
  },
  "2.5Y 2.5/1": {
    "name": "Black",
    "hex": "#3C3B2E",
    "description": "2.5Y 2.5/1 - Black (with faint olive/yellow cast). Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck or peat) or material like charcoal, with the faintest hint of olive or yellow.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained", "Acidic if peat"]
  },
  "5Y 8/1": {
    "name": "White",
    "hex": "#EBE5E1",
    "description": "5Y 8/1 - White with a faint yellow cast. Very high value, very low chroma. Typically indicates light-colored parent material such as chalk, marl, diatomaceous earth, or highly bleached sand with minimal iron or organic influence.",
    "properties": ["Very Low Organic Matter", "Parent Material Dominated (Calcareous/Siliceous/Sandy)", "Minimal Iron Staining (Yellowish)", "High Reflectance"]
  },
  "5Y 8/2": {
    "name": "White",
    "hex": "#E7DBCE",
    "description": "5Y 8/2 - White with a pale yellow tint. Very high value, low chroma. A very light, off-white with a more noticeable pale yellow tint from disseminated yellowish iron oxides (like goethite) or other pale yellow minerals.",
    "properties": ["Low Organic Matter", "Disseminated Yellowish Minerals/Goethite", "Well-Aerated if not gleyed", "Light Mineral Base"]
  },
  "5Y 8/3": {
    "name": "Pale Yellow",
    "hex": "#E4D1C2",
    "description": "5Y 8/3 - Pale Yellow. High value, low to moderate chroma. A light, clear pale yellow, indicating some presence of yellowish iron compounds like goethite in a light matrix.",
    "properties": ["Some Goethite Presence", "Good Aeration", "Low Organic Matter", "Often in well-drained loamy or sandy soils, or specific eolian deposits"]
  },
  "5Y 8/4": {
    "name": "Pale Yellow",
    "hex": "#E1C7B6",
    "description": "5Y 8/4 - Pale Yellow. High value, moderate chroma. A clearer light yellow, showing a more significant concentration of yellowish iron oxides (goethite) within the light-colored soil material.",
    "properties": ["Moderate Goethite Presence", "Well-Drained", "Oxidizing Conditions", "Low Organic Matter"]
  },
  "5Y 8/6": {
    "name": "Yellow",
    "hex": "#DCB098",
    "description": "5Y 8/6 - Yellow. High value, moderate to high chroma. A light, distinct yellow, indicating a higher concentration of yellowish iron oxides like goethite.",
    "properties": ["Higher Goethite Content", "Well-Oxidized", "Good Drainage", "Low Organic Matter", "Can indicate specific mineralogy or parent material like some tuffs"]
  },
  "5Y 8/8": {
    "name": "Yellow",
    "hex": "#D79C7A",
    "description": "5Y 8/8 - Yellow. High value, high chroma. A strong, bright light yellow. Suggests very high concentration of specific yellowish iron minerals; less common in most agricultural soils, more typical of certain geological deposits or intense weathering products under specific conditions.",
    "properties": ["Concentrated Yellow Iron Oxides (Goethite)", "Very Well-Aerated", "Specific Geological Context", "Rare for typical topsoils"]
  },
  "5Y 7/1": {
    "name": "Light Gray",
    "hex": "#D2CBC1",
    "description": "5Y 7/1 - Light Gray with a faint yellow tint. High value, very low chroma. Predominantly gray with a barely discernible yellowish or pale olive tint. Often indicates low organic matter and minimal iron oxide development, or a gleyed condition if mottles are present.",
    "properties": ["Low Organic Matter", "Mineral Dominated", "Slight Yellowish/Olive Tint", "Parent Material Color, Leached Horizon (E), or Gleyed"]
  },
  "5Y 7/2": {
    "name": "Light Gray",
    "hex": "#CBBCB1",
    "description": "5Y 7/2 - Light Gray with a pale yellow cast. High value, low chroma. A light grayish-yellow or pale olive gray, indicating some yellowish iron oxide (goethite) presence or early gleying.",
    "properties": ["Some Yellowish Iron Oxides (Goethite)", "Low Organic Matter", "Generally Well-Drained or Seasonally Wet"]
  },
  "5Y 7/3": {
    "name": "Pale Yellow",
    "hex": "#C5AF9F",
    "description": "5Y 7/3 - Pale Yellow. High value, low-moderate chroma. A pale yellowish-brown or light olive gray, characteristic of soils with modest amounts of goethite and low organic matter.",
    "properties": ["Modest Goethite Content", "Good Aeration or slightly impeded drainage", "Low Organic Matter", "Can be found in loess, alluvium, or weathered parent materials"]
  },
  "5Y 7/4": {
    "name": "Pale Yellow",
    "hex": "#C0A28E",
    "description": "5Y 7/4 - Pale Yellow. High value, moderate chroma. A distinct light yellowish-brown or pale olive, showing a clear presence of goethite in a well-aerated environment or specific gley conditions without strong reduction.",
    "properties": ["Clear Goethite Presence", "Well-Drained or moderately so", "Oxidizing Environment for goethite formation"]
  },
  "5Y 7/6": {
    "name": "Yellow",
    "hex": "#B98E74",
    "description": "5Y 7/6 - Yellow. High value, moderate to high chroma. A brighter, yellowish-brown or olive yellow indicating a more substantial accumulation of goethite or other yellow iron pigments.",
    "properties": ["Substantial Goethite", "Good Aeration and Drainage", "Often from Weathering of iron-bearing minerals"]
  },
  "5Y 7/8": {
    "name": "Yellow",
    "hex": "#B27A57",
    "description": "5Y 7/8 - Yellow. High value, high chroma. A strong and clear yellow or yellowish-orange brown, signifying abundant and relatively pure goethite or similar yellowish iron oxides.",
    "properties": ["Abundant Yellow Iron Oxides (Goethite)", "Well-Oxidized", "May indicate specific parent materials or conditions favoring intense goethite formation"]
  },
  "5Y 6/1": {
    "name": "Gray",
    "hex": "#B3AE9B",
    "description": "5Y 6/1 - Gray with a yellow or olive tint. Moderate value, very low chroma. A neutral gray appearance with a subtle yellowish or olive influence, often from parent material or gleying processes.",
    "properties": ["Predominantly Mineral", "Low Organic Matter", "Subtle Yellowish/Olive Tinge (Goethite or Gleying)", "Often indicates impeded drainage if gleyed"]
  },
  "5Y 6/2": {
    "name": "Light Olive Gray",
    "hex": "#ABA08C",
    "description": "5Y 6/2 - Light Olive Gray. Moderate value, low chroma. A muted brownish-gray with a distinct yellowish or olive hue, indicating some goethite or other Fe-bearing minerals under moderately reduced or parent material conditions.",
    "properties": ["Some Goethite/Olivine Weathering products", "Mixed Mineral Soil", "Moderate to Impeded Drainage", "Low to Moderate Organic Matter"]
  },
  "5Y 6/3": {
    "name": "Pale Olive",
    "hex": "#A4917C",
    "description": "5Y 6/3 - Pale Olive or Light Yellowish Brown. Moderate value, low-moderate chroma. A lighter, less saturated yellowish-brown or olive color, where goethite is present.",
    "properties": ["Goethite Presence", "Fair Drainage or seasonally wet", "Common in C horizons, Bw horizons or gleyed soils"]
  },
  "5Y 6/4": {
    "name": "Light Yellowish Brown",
    "hex": "#9D826B",
    "description": "5Y 6/4 - Light Yellowish Brown or Pale Olive. Moderate value, moderate chroma. A more distinct light yellowish-brown or olive brown, indicating a good presence of goethite.",
    "properties": ["Good Goethite Presence", "Well-Aerated or moderately drained", "Often in Bw or BC horizons"]
  },
  "5Y 6/6": {
    "name": "Olive Yellow",
    "hex": "#956F51",
    "description": "5Y 6/6 - Olive Yellow. Moderate value, moderate to high chroma. A clear olive yellow or brownish-yellow, typical of soils with significant goethite accumulation.",
    "properties": ["Significant Goethite Accumulation", "Well-Drained", "Often in cambic horizons or well-oxidized parent materials containing Fe"]
  },
  "5Y 6/8": {
    "name": "Olive Yellow",
    "hex": "#8E5B38",
    "description": "5Y 6/8 - Olive Yellow. Moderate value, high chroma. A strong olive yellow or brownish-yellow, indicating a high concentration of goethite, common in well-weathered soils or specific sedimentary rocks.",
    "properties": ["High Goethite Concentration", "Intense Weathering Product (for goethite)", "Well-Oxidized Environment"]
  },
  "5Y 5/1": {
    "name": "Gray",
    "hex": "#817E6B",
    "description": "5Y 5/1 - Gray (distinct olive gray). Moderate to low value, very low chroma. A darker gray with a distinct olive or yellowish hue. Can indicate organic matter accumulation or prominent gleying conditions.",
    "properties": ["Some Organic Matter or Gleyed", "Distinct Olive/Yellowish Hue", "Impeded Drainage common if gleyed", "Reduced iron forms possible"]
  },
  "5Y 5/2": {
    "name": "Olive Gray",
    "hex": "#7D715E",
    "description": "5Y 5/2 - Olive Gray. Moderate to low value, low chroma. A dull, weak olive brown or dark olive gray, suggesting mixing with organic matter or gleying.",
    "properties": ["Organic Matter Mixing or Gleying", "Moderate Aeration or periodically anaerobic", "Often in A horizons or gleyed subsoils"]
  },
  "5Y 5/3": {
    "name": "Olive",
    "hex": "#7B6653",
    "description": "5Y 5/3 - Olive or Olive Brown. Moderate to low value, low-moderate chroma. A common soil color with a distinct olive/yellow hue, typical of A or B horizons with moderate goethite and some organic incorporation, or gleying.",
    "properties": ["Moderate Goethite (Olive tone)", "Some Organic Incorporation or Gleying", "Generally Good Soil Structure if not compacted"]
  },
  "5Y 5/4": {
    "name": "Olive",
    "hex": "#795D49",
    "description": "5Y 5/4 - Olive or Olive Brown. Moderate to low value, moderate chroma. Indicates well-aerated conditions with good goethite presence yielding an olive brown, or specific gleyed conditions.",
    "properties": ["Well-Aerated or Gleyed", "Good Goethite Presence (Olive)", "Can be fertile depending on other factors"]
  },
  "5Y 5/6": {
    "name": "Yellowish Brown",
    "hex": "#764F37",
    "description": "5Y 5/6 - Yellowish Brown (with olive cast). Moderate to low value, moderate to high chroma. A clear, strong yellowish-brown or olive brown, indicating high amounts of goethite and good oxidation.",
    "properties": ["High Goethite Content", "Good Drainage and Aeration", "Often found in well-developed B horizons"]
  },
  "5Y 4/1": {
    "name": "Dark Gray",
    "hex": "#615E4F",
    "description": "5Y 4/1 - Dark Gray (dark olive gray). Low value, very low chroma. Dark color, often due to significant organic matter content, with a distinct olive or yellowish hue from underlying iron oxides, minerals or gleying.",
    "properties": ["Significant Organic Matter or Gleyed", "Distinct Goethite/Olive Mineral Presence", "Moisture Retentive", "Fertile A-horizons or gleyed soils"]
  },
  "5Y 4/2": {
    "name": "Dark Olive Gray",
    "hex": "#5F5444",
    "description": "5Y 4/2 - Dark Olive Gray. Low value, low chroma. A dark soil with a muted olive brown color, suggesting a mix of considerable organic matter and some goethite/olivine weathering products, or gleying.",
    "properties": ["Considerable Organic Matter or Gleyed", "Some Goethite/Olivine Influence", "Typically Fertile if not strongly gleyed", "Good Moisture Holding Capacity"]
  },
  "5Y 4/3": {
    "name": "Olive Brown",
    "hex": "#5F4D3B",
    "description": "5Y 4/3 - Olive Brown (dark). Low value, low-moderate chroma. Common in fertile topsoils (A horizons) or gleyed soils where organic matter and goethite (giving olive tone) contribute significantly to the color.",
    "properties": ["Good Organic Content or Gleyed", "Moderate Goethite (Olive)", "Productive or indicative of wetness"]
  },
  "5Y 4/4": {
    "name": "Dark Yellowish Brown",
    "hex": "#604631",
    "description": "5Y 4/4 - Dark Yellowish Brown (dark olive brown). Low value, moderate chroma. A rich, dark olive brown or dark yellowish-brown indicating good aeration and significant goethite, or organic accumulation.",
    "properties": ["Well-Aerated or Organic Rich", "Productive Soil Horizon", "High Goethite Content if not purely organic"]
  },
  "5Y 3/1": {
    "name": "Very Dark Gray",
    "hex": "#4B493B",
    "description": "5Y 3/1 - Very Dark Gray (very dark olive gray). Very low value, very low chroma. Very dark, typically organic-rich soil with a subtle hint of underlying olive/yellowish minerals, or strongly gleyed.",
    "properties": ["High Organic Matter Content (Humus) or Strongly Gleyed", "Slight Olive/Yellowish Mineral Influence", "Often in moist to waterlogged conditions", "Very Fertile if drained, or wetland indicator"]
  },
  "5Y 3/2": {
    "name": "Very Dark Grayish Brown",
    "hex": "#4A4131",
    "description": "5Y 3/2 - Very Dark Grayish Brown (very dark olive brown). Very low value, low chroma. A very dark, muted olive brown. High organic matter content strongly influences this color, mixed with some goethite, or gleyed.",
    "properties": ["Very High Organic Matter or Gleyed", "Goethite Present but Masked (Olive tone)", "Moisture Retentive", "Highly Fertile or wetland soil"]
  },
  "5Y 2.5/1": {
    "name": "Black",
    "hex": "#3C3B2F",
    "description": "5Y 2.5/1 - Black with a faint olive/yellow cast. Extremely low value (nearly black), very low chroma. Predominantly organic soil (muck or peat) or material like charcoal, with the faintest hint of olive or yellow.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Charcoal may be present", "Often Waterlogged/Anaerobic", "Very Poorly Drained", "Acidic if peat"]
  },
  "5Y 2.5/2": {
    "name": "Black",
    "hex": "#3B3328",
    "description": "5Y 2.5/2 - Black (with olive brown cast). Extremely low value, low chroma. Very dark soil where organic matter is dominant, but with a noticeable olive brown influence from finely disseminated goethite or other minerals.",
    "properties": ["Dominant Organic Matter", "Noticeable Goethite/Olive Brown Influence", "Poor Drainage Common", "High Potential Fertility if drained"]
  },
  "10Y 6/2": {
    "name": "Light Olive Gray",
    "hex": "#A7A18C",
    "description": "10Y 6/2 - Light Olive Gray. A light gray with a distinct yellowish-olive hue. Often indicates moderately gleyed conditions or parent materials rich in minerals that weather to olive colors.",
    "properties": ["Gleyed or specific parent material", "Moderate to impeded drainage", "Low to moderate organic matter", "Reduced iron conditions possible"]
  },
  "10Y 5/2": {
    "name": "Olive Gray",
    "hex": "#8C8675",
    "description": "10Y 5/2 - Olive Gray. A grayish soil with a clear olive or yellowish tint. Typically associated with gleying processes in subsoils or soils with high water tables.",
    "properties": ["Gleyed conditions", "Poor aeration", "Impeded drainage", "Accumulation of organic matter possible if not too acidic"]
  },
  "10Y 4/2": {
    "name": "Dark Olive Gray",
    "hex": "#716D5E",
    "description": "10Y 4/2 - Dark Olive Gray. A dark gray soil with a noticeable olive or yellowish hue. Indicates significant gleying and reducing conditions, often with higher organic matter content.",
    "properties": ["Strongly gleyed", "Anaerobic conditions", "High water table", "Organic matter accumulation"]
  },
  "10Y 3/2": {
    "name": "Very Dark Olive Gray",
    "hex": "#565347",
    "description": "10Y 3/2 - Very Dark Olive Gray. A very dark, almost black soil with a faint olive or yellowish hue. Usually found in very organic-rich, waterlogged environments like peat or muck with gleying.",
    "properties": ["Very high organic matter (peat/muck)", "Waterlogged", "Strongly reducing conditions", "Gleyed"]
  },
  "10Y 6/4": {
    "name": "Pale Olive",
    "hex": "#A09572",
    "description": "10Y 6/4 - Pale Olive or Light Yellowish Brown. A light olive or yellowish-brown with moderate chroma. Can indicate well-drained soils with goethite, or moderately gleyed soils where some yellow/olive color persists.",
    "properties": ["Goethite presence or moderate gleying", "Moderate drainage", "Low to moderate organic matter", "Mineral soil"]
  },
  "10Y 5/4": {
    "name": "Olive",
    "hex": "#857B60",
    "description": "10Y 5/4 - Olive or Yellowish Brown. A distinct olive or yellowish-brown color. Common in B-horizons of some soils (e.g., Alfisols, Mollisols) with good goethite content or specific parent materials.",
    "properties": ["Goethite rich", "Well to moderately drained", "Moderate organic matter possible in A horizons", "Fertile depending on other factors"]
  },
  "10Y 4/4": {
    "name": "Dark Olive",
    "hex": "#6A604B",
    "description": "10Y 4/4 - Dark Olive or Dark Yellowish Brown. A dark olive or dark yellowish-brown. Indicates soils with significant goethite and some organic matter, or specific mineralogy.",
    "properties": ["Significant goethite", "Organic matter accumulation", "Good aeration if not gleyed", "Often fertile"]
  },
  "10Y 3/4": {
    "name": "Very Dark Olive",
    "hex": "#504A3A",
    "description": "10Y 3/4 - Very Dark Olive. A very dark olive brown. High in organic matter with a strong influence from goethite or other olive-colored minerals.",
    "properties": ["High organic matter", "Goethite or specific minerals", "Moist conditions", "Fertile"]
  },
  "5GY 6/2": {
    "name": "Light Grayish Olive",
    "hex": "#9BA08C",
    "description": "5GY 6/2 - Light Grayish Olive. A light gray with a distinct greenish-yellow (olive) hue. Usually indicates gleyed soils where reduced iron compounds impart greenish colors.",
    "properties": ["Gleyed", "Reduced iron compounds", "Poor aeration", "Impeded drainage", "Often found in wetlands or subsoils with high water table"]
  },
  "5GY 5/2": {
    "name": "Grayish Olive",
    "hex": "#818675",
    "description": "5GY 5/2 - Grayish Olive. An olive gray color with a clear greenish-yellow influence. Typical of gley soils under prolonged waterlogging and reducing conditions.",
    "properties": ["Strongly gleyed", "Anaerobic conditions", "Reduced iron (ferrous)", "Waterlogged", "May have specific clay mineralogy (e.g., smectites)"]
  },
  "5GY 4/2": {
    "name": "Dark Grayish Olive",
    "hex": "#676D5E",
    "description": "5GY 4/2 - Dark Grayish Olive. A dark olive gray with a greenish-yellow cast. Indicates intense gleying and reducing conditions, often with significant organic matter accumulation.",
    "properties": ["Intensely gleyed", "High organic matter", "Very poor drainage", "Saturated conditions common"]
  },
  "5GY 3/2": {
    "name": "Very Dark Grayish Olive",
    "hex": "#4D5347",
    "description": "5GY 3/2 - Very Dark Grayish Olive. A very dark, almost black soil with a faint greenish-yellow or olive hue. Usually found in organic-rich, heavily gleyed environments like swamps or bogs.",
    "properties": ["Very high organic matter (peat/muck)", "Permanently waterlogged", "Strongly reducing, anoxic", "Sulfidic materials possible if S is present"]
  },
  "5GY 6/4": {
    "name": "Pale Olive",
    "hex": "#949672",
    "description": "5GY 6/4 - Pale Olive. A light olive with a moderate greenish-yellow chroma. Can occur in gleyed soils that are not extremely reduced or in soils with minerals like glauconite or chlorite near the surface.",
    "properties": ["Gleyed or specific mineralogy (glauconite/chlorite)", "Impeded to poor drainage", "Low to moderate organic matter if mineral origin", "Reduced conditions"]
  },
  "5GY 5/4": {
    "name": "Olive",
    "hex": "#7A7D60",
    "description": "5GY 5/4 - Olive. A distinct olive color with a greenish-yellow hue. Often indicates gleying, but can also be associated with weathering of certain parent materials (e.g., those rich in olivine or glauconite).",
    "properties": ["Gleying or parent material influence (olivine, glauconite)", "Poor to moderate drainage", "Reduced iron forms present", "Variable organic matter"]
  },
  "5GY 4/4": {
    "name": "Dark Olive",
    "hex": "#60634B",
    "description": "5GY 4/4 - Dark Olive. A dark olive with a clear greenish-yellow hue. Suggests gleyed conditions with some organic matter, or parent rocks rich in dark green minerals like chlorite or amphibole.",
    "properties": ["Gleyed with organic matter or dark green minerals", "Poor drainage", "Reducing environment", "Often fertile if pH is not too low"]
  },
  "5GY 3/4": {
    "name": "Very Dark Olive",
    "hex": "#474D3A",
    "description": "5GY 3/4 - Very Dark Olive. A very dark olive soil, nearly black, with a greenish-yellow influence. Typically very high in organic matter and strongly gleyed, or rich in dark ferromagnesian minerals.",
    "properties": ["Very high organic matter and strongly gleyed, or ferromagnesian rich", "Waterlogged", "Strongly reducing", "Often acidic (e.g., Histosols)"]
  },
  "N 8/": {
    "name": "White (Gleyed)",
    "hex": "#E3E4E2",
    "description": "N 8/ - White (Neutral Gley). A very light gray, almost white, indicating intense reduction and leaching of coloring agents under waterlogged conditions. Little to no iron influence visible.",
    "properties": ["Intense Gleying", "Strongly Reducing Conditions", "Very Poor Drainage", "Leached of Iron", "High Water Table", "Low Organic Matter (unless parent material is organic)"]
  },
  "N 7/": {
    "name": "Light Gray (Gleyed)",
    "hex": "#CACBC9",
    "description": "N 7/ - Light Gray (Neutral Gley). A light gray color typical of gleyed soils where iron has been largely reduced and removed or transformed to colorless forms.",
    "properties": ["Gleyed", "Reducing Conditions", "Poor Drainage", "Low Iron Influence (reduced/removed)", "Often in Aquepts, Aqualfs, etc."]
  },
  "N 6/": {
    "name": "Gray (Gleyed)",
    "hex": "#B1B2B0",
    "description": "N 6/ - Gray (Neutral Gley). A medium gray gley color, indicating persistent water saturation and anaerobic conditions.",
    "properties": ["Persistent Saturation", "Anaerobic", "Poor Aeration", "Reduced Iron", "Hydric Soil Indicator"]
  },
  "N 5/": {
    "name": "Gray (Gleyed)",
    "hex": "#989997",
    "description": "N 5/ - Gray (Neutral Gley). A distinct gray, common in gleyed horizons (Bg or Cg horizons) of poorly drained soils.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Reduced Conditions", "Often Sticky and Plastic Clay"]
  },
  "N 4/": {
    "name": "Dark Gray (Gleyed)",
    "hex": "#7F807E",
    "description": "N 4/ - Dark Gray (Neutral Gley). A dark gray gley color, often indicating accumulation of some organic matter under reducing, waterlogged conditions.",
    "properties": ["Organic Matter Accumulation (possible)", "Waterlogged", "Strongly Reduced", "Low Oxygen"]
  },
  "N 3/": {
    "name": "Dark Gray (Gleyed)",
    "hex": "#656664",
    "description": "N 3/ - Dark Gray (Neutral Gley), approaching black. Very dark gray, typical of gleyed soils with significant organic matter or specific dark-colored reduced minerals.",
    "properties": ["High Organic Matter or Dark Reduced Minerals", "Very Poor Drainage", "Anoxic Conditions", "Often in Histosols or gleyed Mollisols"]
  },
  "N 2.5/": {
    "name": "Black (Gleyed)",
    "hex": "#565755",
    "description": "N 2.5/ - Black (Neutral Gley). A very dark gray to black gley color, usually indicates high organic matter content under permanent saturation and reducing conditions.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Anoxic", "Sulfidic materials possible"]
  },
  "10Y 8/1": {
    "name": "White (Gleyed, Yellow Hue)",
    "hex": "#E4E4DE",
    "description": "10Y 8/1 - White (10Y Gley). Very light gray with a faint yellowish cast. Indicates intense gleying with slight residual yellowish mineral influence.",
    "properties": ["Intense Gleying", "Faint Yellowish Cast", "Very Poor Drainage", "Reduced Conditions"]
  },
  "10Y 7/1": {
    "name": "Light Gray (Gleyed, Yellow Hue)",
    "hex": "#CBCBC3",
    "description": "10Y 7/1 - Light Gray (10Y Gley). Light gray with a yellowish hue, typical of gleyed soils with some influence from minerals weathering to yellow under reducing conditions.",
    "properties": ["Gleyed", "Yellowish Hue", "Poor Drainage", "Reduced Iron"]
  },
  "10Y 6/1": {
    "name": "Gray (Gleyed, Yellow Hue)",
    "hex": "#B1B2A9",
    "description": "10Y 6/1 - Gray (10Y Gley). Medium gray with a yellowish or pale olive hue, indicating persistent water saturation.",
    "properties": ["Persistent Saturation", "Yellowish/Pale Olive Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "10Y 5/1": {
    "name": "Gray (Gleyed, Yellow Hue)",
    "hex": "#98998F",
    "description": "10Y 5/1 - Gray (10Y Gley). Gray with a distinct yellowish or olive hue, common in gleyed horizons.",
    "properties": ["Poorly Drained", "Gleyed Horizon (Bg, Cg)", "Olive/Yellowish Hue", "Reduced Iron Compounds"]
  },
  "10Y 4/1": {
    "name": "Dark Gray (Gleyed, Yellow Hue)",
    "hex": "#7F8077",
    "description": "10Y 4/1 - Dark Gray (10Y Gley). Dark gray gley color with a yellowish or olive cast, often with some organic matter.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Yellowish/Olive Cast", "Strongly Reduced"]
  },
  "10Y 3/1": {
    "name": "Very Dark Gray (Gleyed, Yellow Hue)",
    "hex": "#65665D",
    "description": "10Y 3/1 - Very Dark Gray (10Y Gley). Very dark gray, with a faint yellowish or olive hue. High organic matter under gleyed conditions.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Faint Yellowish/Olive Hue", "Anoxic"]
  },
  "10Y 2.5/1": {
    "name": "Black (Gleyed, Yellow Hue)",
    "hex": "#56574E",
    "description": "10Y 2.5/1 - Black (10Y Gley). Very dark gray to black with a subtle yellowish or olive hue, typically organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Subtle Yellowish/Olive Hue", "Anoxic"]
  },
  "5GY 8/1": {
    "name": "White (Gleyed, Green-Yellow Hue)",
    "hex": "#E2E4DE",
    "description": "5GY 8/1 - White (5GY Gley). Very light gray with a faint greenish-yellow cast. Indicates intense gleying with slight residual greenish-yellow mineral influence.",
    "properties": ["Intense Gleying", "Faint Greenish-Yellow Cast", "Very Poor Drainage", "Reduced Conditions"]
  },
  "5GY 7/1": {
    "name": "Light Gray (Gleyed, Green-Yellow Hue)",
    "hex": "#C9CBC3",
    "description": "5GY 7/1 - Light Gray (5GY Gley). Light gray with a greenish-yellow hue, typical of gleyed soils with reduced iron forming greenish compounds.",
    "properties": ["Gleyed", "Greenish-Yellow Hue", "Poor Drainage", "Reduced Iron (Fe²⁺)"]
  },
  "5GY 6/1": {
    "name": "Gray (Gleyed, Green-Yellow Hue)",
    "hex": "#B0B2A9",
    "description": "5GY 6/1 - Gray (5GY Gley). Medium gray with a greenish-yellow or pale olive hue, indicating persistent water saturation and reduction.",
    "properties": ["Persistent Saturation", "Greenish-Yellow/Pale Olive Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "5GY 5/1": {
    "name": "Olive Gray (Gleyed)",
    "hex": "#97998F",
    "description": "5GY 5/1 - Olive Gray (5GY Gley). Gray with a distinct greenish-yellow or olive hue, common in gleyed horizons.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Olive/Greenish-Yellow Hue", "Reduced Iron Compounds"]
  },
  "5GY 4/1": {
    "name": "Dark Olive Gray (Gleyed)",
    "hex": "#7E8077",
    "description": "5GY 4/1 - Dark Olive Gray (5GY Gley). Dark gray gley color with a greenish-yellow or olive cast, often with some organic matter.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Greenish-Yellow/Olive Cast", "Strongly Reduced"]
  },
  "5GY 3/1": {
    "name": "Very Dark Olive Gray (Gleyed)",
    "hex": "#64665D",
    "description": "5GY 3/1 - Very Dark Olive Gray (5GY Gley). Very dark gray, with a faint greenish-yellow or olive hue. High organic matter under gleyed conditions.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Faint Greenish-Yellow/Olive Hue", "Anoxic"]
  },
  "5GY 2.5/1": {
    "name": "Black (Gleyed, Green-Yellow Hue)",
    "hex": "#55574E",
    "description": "5GY 2.5/1 - Black (5GY Gley). Very dark gray to black with a subtle greenish-yellow or olive hue, typically organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Subtle Greenish-Yellow/Olive Hue", "Anoxic"]
  },
  "10GY 8/1": {
    "name": "White (Gleyed, Green-Yellow Hue)",
    "hex": "#E2E4DB",
    "description": "10GY 8/1 - White (10GY Gley). Very light gray with a stronger greenish-yellow cast than 5GY. Intense gleying.",
    "properties": ["Intense Gleying", "Stronger Greenish-Yellow Cast", "Very Poor Drainage", "Reduced Conditions"]
  },
  "10GY 7/1": {
    "name": "Light Gray (Gleyed, Green-Yellow Hue)",
    "hex": "#C9CBC0",
    "description": "10GY 7/1 - Light Gray (10GY Gley). Light gray with a noticeable greenish-yellow hue. Gleyed soil.",
    "properties": ["Gleyed", "Noticeable Greenish-Yellow Hue", "Poor Drainage", "Reduced Iron (Fe²⁺)"]
  },
  "10GY 6/1": {
    "name": "Gray (Gleyed, Green-Yellow Hue)",
    "hex": "#B0B2A6",
    "description": "10GY 6/1 - Gray (10GY Gley). Medium gray with a distinct greenish-yellow hue. Persistent saturation.",
    "properties": ["Persistent Saturation", "Distinct Greenish-Yellow Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "10GY 5/1": {
    "name": "Olive Gray (Gleyed)",
    "hex": "#97998C",
    "description": "10GY 5/1 - Olive Gray (10GY Gley). Gray with a strong greenish-yellow or olive hue. Common in gleyed horizons.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Strong Olive/Greenish-Yellow Hue", "Reduced Iron Compounds"]
  },
  "10GY 4/1": {
    "name": "Dark Olive Gray (Gleyed)",
    "hex": "#7E8074",
    "description": "10GY 4/1 - Dark Olive Gray (10GY Gley). Dark gray gley color with a strong greenish-yellow or olive cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Strong Greenish-Yellow/Olive Cast", "Strongly Reduced"]
  },
  "10GY 3/1": {
    "name": "Very Dark Gray (Gleyed, Green-Yellow Hue)",
    "hex": "#64665A",
    "description": "10GY 3/1 - Very Dark Gray (10GY Gley). Very dark gray, with a greenish-yellow or olive hue. High organic matter.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Greenish-Yellow/Olive Hue", "Anoxic"]
  },
  "10GY 2.5/1": {
    "name": "Black (Gleyed, Green-Yellow Hue)",
    "hex": "#55574B",
    "description": "10GY 2.5/1 - Black (10GY Gley). Very dark gray to black with a greenish-yellow or olive hue. Organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Greenish-Yellow/Olive Hue", "Anoxic"]
  },
  "5G 8/1": {
    "name": "White (Gleyed, Green Hue)",
    "hex": "#E1E4DC",
    "description": "5G 8/1 - White (5G Gley /1 Chroma). Very light gray with a faint green cast. Intense gleying, reduced iron compounds imparting greenish color.",
    "properties": ["Intense Gleying", "Faint Green Cast", "Very Poor Drainage", "Reduced Iron (Fe²⁺)"]
  },
  "5G 7/1": {
    "name": "Light Greenish Gray (Gleyed)",
    "hex": "#C8CBC1",
    "description": "5G 7/1 - Light Greenish Gray (5G Gley /1 Chroma). Light gray with a definite green hue. Gleyed soil.",
    "properties": ["Gleyed", "Definite Green Hue", "Poor Drainage", "Reduced Iron (Fe²⁺)"]
  },
  "5G 6/1": {
    "name": "Greenish Gray (Gleyed)",
    "hex": "#AFB2A7",
    "description": "5G 6/1 - Greenish Gray (5G Gley /1 Chroma). Medium gray with a distinct green hue. Persistent saturation.",
    "properties": ["Persistent Saturation", "Distinct Green Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "5G 5/1": {
    "name": "Greenish Gray (Gleyed)",
    "hex": "#96998D",
    "description": "5G 5/1 - Greenish Gray (5G Gley /1 Chroma). Gray with a strong green hue. Common in gleyed horizons with specific Fe²⁺ minerals.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Strong Green Hue", "Reduced Iron Minerals"]
  },
  "5G 4/1": {
    "name": "Dark Greenish Gray (Gleyed)",
    "hex": "#7D8075",
    "description": "5G 4/1 - Dark Greenish Gray (5G Gley /1 Chroma). Dark gray gley color with a strong green cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Strong Green Cast", "Strongly Reduced"]
  },
  "5G 3/1": {
    "name": "Very Dark Greenish Gray (Gleyed)",
    "hex": "#63665B",
    "description": "5G 3/1 - Very Dark Greenish Gray (5G Gley /1 Chroma). Very dark gray, with a green hue. High organic matter.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Green Hue", "Anoxic"]
  },
  "5G 2.5/1": {
    "name": "Black (Gleyed, Green Hue)",
    "hex": "#54574C",
    "description": "5G 2.5/1 - Black (5G Gley /1 Chroma). Very dark gray to black with a green hue. Organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Green Hue", "Anoxic"]
  },
  "5G 8/2": {
    "name": "Pale Green (Gleyed)",
    "hex": "#D8E0D4",
    "description": "5G 8/2 - Pale Green (5G Gley /2 Chroma). Very light gray with a more noticeable green cast due to higher chroma. Intense gleying.",
    "properties": ["Intense Gleying", "Noticeable Green Cast", "Very Poor Drainage", "Reduced Iron (Fe²⁺)"]
  },
  "5G 7/2": {
    "name": "Light Green (Gleyed)",
    "hex": "#BFCCC0",
    "description": "5G 7/2 - Light Green (5G Gley /2 Chroma). Light gray with a clear and definite green hue. Gleyed soil.",
    "properties": ["Gleyed", "Clear Green Hue", "Poor Drainage", "Reduced Iron (Fe²⁺)"]
  },
  "5G 6/2": {
    "name": "Greenish Gray (Gleyed)",
    "hex": "#A6B2AA",
    "description": "5G 6/2 - Greenish Gray (5G Gley /2 Chroma). Medium gray with a distinct and stronger green hue. Persistent saturation.",
    "properties": ["Persistent Saturation", "Stronger Green Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "5G 5/2": {
    "name": "Green (Gleyed)",
    "hex": "#8C9990",
    "description": "5G 5/2 - Green (5G Gley /2 Chroma). Grayish green with a strong green hue. Common in gleyed horizons with specific Fe²⁺ minerals like green rust.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Strong Green Hue (may indicate green rust)", "Reduced Iron Minerals"]
  },
  "5G 4/2": {
    "name": "Dark Green (Gleyed)",
    "hex": "#738078",
    "description": "5G 4/2 - Dark Green (5G Gley /2 Chroma). Dark gray gley color with a strong and clear green cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Strong and Clear Green Cast", "Strongly Reduced"]
  },
  "5G 3/2": {
    "name": "Very Dark Green (Gleyed)",
    "hex": "#59665E",
    "description": "5G 3/2 - Very Dark Green (5G Gley /2 Chroma). Very dark gray, approaching black, with a distinct green hue. High organic matter.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Distinct Green Hue", "Anoxic"]
  },
  "5G 2.5/2": {
    "name": "Black (Gleyed, Strong Green Hue)",
    "hex": "#4A574F",
    "description": "5G 2.5/2 - Black (5G Gley /2 Chroma). Very dark gray to black with a strong green hue. Organic-rich gley soil with prominent green coloration from reduced iron compounds.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Strong Green Hue", "Anoxic", "May have vivianite if phosphorus is present"]
  },
  "10G 8/1": {
    "name": "White (Greenish Gley)",
    "hex": "#E1E4DE",
    "description": "10G 8/1 - White with a faint very pale green cast. Very high value, low chroma. Indicates intense gleying and reduction, with minimal pigmenting compounds remaining or present as pale green reduced forms.",
    "properties": ["Intense Gleying", "Strongly Reducing Conditions", "Very Poor Drainage", "Leached or Transformed Iron", "High Water Table"]
  },
  "10G 7/1": {
    "name": "Light Greenish Gray (Gleyed)",
    "hex": "#C8CBCA",
    "description": "10G 7/1 - Light Greenish Gray. A light gray with a distinct pale green hue, typical of gleyed soils where iron is in a reduced state, forming greenish compounds.",
    "properties": ["Gleyed", "Reduced Iron (Fe²⁺)", "Poor Drainage", "Anaerobic Conditions", "Hydric Soil Indicator"]
  },
  "10G 6/1": {
    "name": "Greenish Gray (Gleyed)",
    "hex": "#AFB2B1",
    "description": "10G 6/1 - Greenish Gray. A medium gray with a noticeable pale green hue, indicating persistent water saturation and reducing conditions.",
    "properties": ["Persistent Saturation", "Anaerobic", "Poor Aeration", "Greenish Reduced Iron Compounds"]
  },
  "10G 5/1": {
    "name": "Greenish Gray (Gleyed)",
    "hex": "#969998",
    "description": "10G 5/1 - Greenish Gray. A distinct gray with a pale green hue, common in gleyed horizons (Bg or Cg) of poorly drained soils.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Reduced Conditions", "Often Sticky and Plastic Clay Texture"]
  },
  "10G 4/1": {
    "name": "Dark Greenish Gray (Gleyed)",
    "hex": "#7D807F",
    "description": "10G 4/1 - Dark Greenish Gray. A dark gray gley color with a pale green cast, often indicating accumulation of some organic matter under reducing, waterlogged conditions.",
    "properties": ["Organic Matter Accumulation (possible)", "Waterlogged", "Strongly Reduced", "Low Oxygen Environment"]
  },
  "10G 3/1": {
    "name": "Dark Greenish Gray (Gleyed)",
    "hex": "#636665",
    "description": "10G 3/1 - Dark Greenish Gray, approaching black. Very dark gray, typical of gleyed soils with significant organic matter or specific dark-colored reduced minerals with a greenish tint.",
    "properties": ["High Organic Matter or Dark Reduced Minerals", "Very Poor Drainage", "Anoxic Conditions", "Greenish Tint"]
  },
  "10G 2.5/1": {
    "name": "Black (Greenish Gley)",
    "hex": "#545756",
    "description": "10G 2.5/1 - Black with a faint greenish cast. A very dark gray to black gley color, usually indicates high organic matter content under permanent saturation with some greenish reduced compounds.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Anoxic", "Faint Greenish Cast"]
  },
  "5BG 8/1": {
    "name": "White (Blueish Green Gley)",
    "hex": "#E0E4E1",
    "description": "5BG 8/1 - White with a faint pale blue-green cast. Indicates intense gleying. The blue-green hue may be due to specific finely disseminated reduced iron or manganese compounds.",
    "properties": ["Intense Gleying", "Faint Blue-Green Cast", "Very Poor Drainage", "Reduced Conditions"]
  },
  "5BG 7/1": {
    "name": "Light Bluish Gray (Gleyed)",
    "hex": "#C7CBC9",
    "description": "5BG 7/1 - Light Bluish Gray. Light gray with a pale blue-green hue, typical of strongly gleyed soils.",
    "properties": ["Gleyed", "Pale Blue-Green Hue", "Poor Drainage", "Reduced Iron/Manganese Compounds"]
  },
  "5BG 6/1": {
    "name": "Bluish Gray (Gleyed)",
    "hex": "#AEB2B0",
    "description": "5BG 6/1 - Bluish Gray. Medium gray with a noticeable pale blue-green hue. Persistent water saturation.",
    "properties": ["Persistent Saturation", "Pale Blue-Green Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "5BG 5/1": {
    "name": "Bluish Gray (Gleyed)",
    "hex": "#959997",
    "description": "5BG 5/1 - Bluish Gray. Distinct gray with a pale blue-green hue, common in gleyed horizons with reduced minerals.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Pale Blue-Green Hue", "Reduced Mineral Forms"]
  },
  "5BG 4/1": {
    "name": "Dark Bluish Gray (Gleyed)",
    "hex": "#7C807E",
    "description": "5BG 4/1 - Dark Bluish Gray. Dark gray gley color with a pale blue-green cast. Often with organic matter accumulation.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Pale Blue-Green Cast", "Strongly Reduced"]
  },
  "5BG 3/1": {
    "name": "Dark Bluish Gray (Gleyed)",
    "hex": "#626664",
    "description": "5BG 3/1 - Dark Bluish Gray. Very dark gray, with a pale blue-green hue. High organic matter under gleyed conditions.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Pale Blue-Green Hue", "Anoxic"]
  },
  "5BG 2.5/1": {
    "name": "Black (Bluish Green Gley)",
    "hex": "#535755",
    "description": "5BG 2.5/1 - Black with a faint blue-green cast. Very dark gray to black, typically organic-rich gley soil with blue-green reduced compounds.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Faint Blue-Green Cast", "Anoxic"]
  },
  "10BG 8/1": {
    "name": "White (Greenish Blue Gley)",
    "hex": "#E0E4E0",
    "description": "10BG 8/1 - White with a faint pale greenish-blue cast. Intense gleying and reduction.",
    "properties": ["Intense Gleying", "Faint Greenish-Blue Cast", "Very Poor Drainage", "Reduced Conditions"]
  },
  "10BG 7/1": {
    "name": "Light Greenish Gray (Gleyed, Bluish)",
    "hex": "#C7CBC8",
    "description": "10BG 7/1 - Light Greenish Gray with a bluish tint. Light gray with a pale greenish-blue hue.",
    "properties": ["Gleyed", "Pale Greenish-Blue Hue", "Poor Drainage", "Reduced Iron/Manganese"]
  },
  "10BG 6/1": {
    "name": "Greenish Gray (Gleyed, Bluish)",
    "hex": "#AEB2AF",
    "description": "10BG 6/1 - Greenish Gray with a bluish tint. Medium gray with a pale greenish-blue hue.",
    "properties": ["Persistent Saturation", "Pale Greenish-Blue Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "10BG 5/1": {
    "name": "Greenish Gray (Gleyed, Bluish)",
    "hex": "#959996",
    "description": "10BG 5/1 - Greenish Gray with a bluish tint. Distinct gray with a pale greenish-blue hue.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Pale Greenish-Blue Hue", "Reduced Mineral Forms"]
  },
  "10BG 4/1": {
    "name": "Dark Greenish Gray (Gleyed, Bluish)",
    "hex": "#7C807D",
    "description": "10BG 4/1 - Dark Greenish Gray with a bluish tint. Dark gray gley color with a pale greenish-blue cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Pale Greenish-Blue Cast", "Strongly Reduced"]
  },
  "10BG 3/1": {
    "name": "Dark Greenish Gray (Gleyed, Bluish)",
    "hex": "#626663",
    "description": "10BG 3/1 - Dark Greenish Gray with a bluish tint. Very dark gray, with a pale greenish-blue hue.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Pale Greenish-Blue Hue", "Anoxic"]
  },
  "10BG 2.5/1": {
    "name": "Black (Greenish Blue Gley)",
    "hex": "#535754",
    "description": "10BG 2.5/1 - Black with a faint greenish-blue cast. Very dark gray to black, organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Faint Greenish-Blue Cast", "Anoxic"]
  },
  "5B 8/1": {
    "name": "White (Bluish Gley)",
    "hex": "#E0E3E2",
    "description": "5B 8/1 - White with a faint pale blue cast. Intense gleying. Blue hues can be due to vivianite (iron phosphate) or other reduced minerals.",
    "properties": ["Intense Gleying", "Faint Pale Blue Cast", "Very Poor Drainage", "Reduced Conditions", "Vivianite possible"]
  },
  "5B 7/1": {
    "name": "Light Bluish Gray (Gleyed)",
    "hex": "#C7CACB",
    "description": "5B 7/1 - Light Bluish Gray. Light gray with a pale blue hue.",
    "properties": ["Gleyed", "Pale Blue Hue", "Poor Drainage", "Reduced Iron/Vivianite"]
  },
  "5B 6/1": {
    "name": "Bluish Gray (Gleyed)",
    "hex": "#AEB1B2",
    "description": "5B 6/1 - Bluish Gray. Medium gray with a pale blue hue.",
    "properties": ["Persistent Saturation", "Pale Blue Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "5B 5/1": {
    "name": "Bluish Gray (Gleyed)",
    "hex": "#959899",
    "description": "5B 5/1 - Bluish Gray. Distinct gray with a pale blue hue.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Pale Blue Hue", "Reduced Mineral Forms (e.g., vivianite)"]
  },
  "5B 4/1": {
    "name": "Dark Bluish Gray (Gleyed)",
    "hex": "#7C7F80",
    "description": "5B 4/1 - Dark Bluish Gray. Dark gray gley color with a pale blue cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Pale Blue Cast", "Strongly Reduced"]
  },
  "5B 3/1": {
    "name": "Dark Bluish Gray (Gleyed)",
    "hex": "#626566",
    "description": "5B 3/1 - Dark Bluish Gray. Very dark gray, with a pale blue hue.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Pale Blue Hue", "Anoxic"]
  },
  "5B 2.5/1": {
    "name": "Black (Bluish Gley)",
    "hex": "#535657",
    "description": "5B 2.5/1 - Black with a faint blue cast. Very dark gray to black, organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Faint Blue Cast", "Anoxic"]
  },
  "10B 8/1": {
    "name": "White (Bluish Gley)",
    "hex": "#E0E3E3",
    "description": "10B 8/1 - White with a clearer pale blue cast than 5B. Intense gleying.",
    "properties": ["Intense Gleying", "Clearer Pale Blue Cast", "Very Poor Drainage", "Reduced Conditions"]
  },
  "10B 7/1": {
    "name": "Light Bluish Gray (Gleyed)",
    "hex": "#C7CACB",
    "description": "10B 7/1 - Light Bluish Gray. Light gray with a more definite pale blue hue.",
    "properties": ["Gleyed", "Definite Pale Blue Hue", "Poor Drainage", "Reduced Iron/Vivianite"]
  },
  "10B 6/1": {
    "name": "Bluish Gray (Gleyed)",
    "hex": "#AEB1B2",
    "description": "10B 6/1 - Bluish Gray. Medium gray with a clear pale blue hue.",
    "properties": ["Persistent Saturation", "Clear Pale Blue Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "10B 5/1": {
    "name": "Bluish Gray (Gleyed)",
    "hex": "#959899",
    "description": "10B 5/1 - Bluish Gray. Distinct gray with a clear pale blue hue.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Clear Pale Blue Hue", "Reduced Mineral Forms (e.g., vivianite)"]
  },
  "10B 4/1": {
    "name": "Dark Bluish Gray (Gleyed)",
    "hex": "#7C7F80",
    "description": "10B 4/1 - Dark Bluish Gray. Dark gray gley color with a clear pale blue cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Clear Pale Blue Cast", "Strongly Reduced"]
  },
  "10B 3/1": {
    "name": "Dark Bluish Gray (Gleyed)",
    "hex": "#626566",
    "description": "10B 3/1 - Dark Bluish Gray. Very dark gray, with a clear pale blue hue.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Clear Pale Blue Hue", "Anoxic"]
  },
  "10B 2.5/1": {
    "name": "Black (Bluish Gley)",
    "hex": "#535657",
    "description": "10B 2.5/1 - Black with a clear blue cast. Very dark gray to black, organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Clear Blue Cast", "Anoxic"]
  },
  "5PB 8/1": {
    "name": "White (Purplish Blue Gley)",
    "hex": "#E1E3E3",
    "description": "5PB 8/1 - White with a faint pale purplish-blue cast. Intense gleying. Purplish-blue hues are less common, may indicate specific manganese or iron-manganese compounds under reduction.",
    "properties": ["Intense Gleying", "Faint Pale Purplish-Blue Cast", "Very Poor Drainage", "Reduced Conditions", "Specific Mn/Fe compounds"]
  },
  "5PB 7/1": {
    "name": "Light Purplish Gray (Gleyed)",
    "hex": "#C8CACB",
    "description": "5PB 7/1 - Light Purplish Gray. Light gray with a pale purplish-blue hue.",
    "properties": ["Gleyed", "Pale Purplish-Blue Hue", "Poor Drainage", "Reduced Compounds"]
  },
  "5PB 6/1": {
    "name": "Purplish Gray (Gleyed)",
    "hex": "#AFB1B2",
    "description": "5PB 6/1 - Purplish Gray. Medium gray with a pale purplish-blue hue.",
    "properties": ["Persistent Saturation", "Pale Purplish-Blue Hue", "Poor Aeration", "Reduced Conditions"]
  },
  "5PB 5/1": {
    "name": "Purplish Gray (Gleyed)",
    "hex": "#969899",
    "description": "5PB 5/1 - Purplish Gray. Distinct gray with a pale purplish-blue hue.",
    "properties": ["Poorly Drained", "Gleyed Horizon", "Pale Purplish-Blue Hue", "Reduced Mineral Forms"]
  },
  "5PB 4/1": {
    "name": "Dark Purplish Gray (Gleyed)",
    "hex": "#7D7F80",
    "description": "5PB 4/1 - Dark Purplish Gray. Dark gray gley color with a pale purplish-blue cast.",
    "properties": ["Organic Matter Influence", "Waterlogged", "Pale Purplish-Blue Cast", "Strongly Reduced"]
  },
  "5PB 3/1": {
    "name": "Dark Purplish Gray (Gleyed)",
    "hex": "#636566",
    "description": "5PB 3/1 - Dark Purplish Gray. Very dark gray, with a pale purplish-blue hue.",
    "properties": ["High Organic Matter", "Very Poor Drainage", "Pale Purplish-Blue Hue", "Anoxic"]
  },
  "5PB 2.5/1": {
    "name": "Black (Purplish Blue Gley)",
    "hex": "#545657",
    "description": "5PB 2.5/1 - Black with a faint purplish-blue cast. Very dark gray to black, organic-rich gley soil.",
    "properties": ["Very High Organic Matter (Muck/Peat)", "Permanently Saturated", "Faint Purplish-Blue Cast", "Anoxic"]
  },
  "N 9.5/": {
    "name": "White",
    "hex": "#F5F5F5",
    "description": "N 9.5/ - White (Neutral). Extremely high value, neutral color. Represents very pure white materials like refined kaolin, chalk, or fresh snow. In soils, could be pure gypsum or carbonate deposits.",
    "properties": ["Very Pure Light Material", "Extremely Low Organic Matter", "High Reflectance", "Often Calcareous, Gypsiferous, or Saline"]
  },
  "N 9/": {
    "name": "White",
    "hex": "#E8E8E8",
    "description": "N 9/ - White (Neutral). High value, neutral color. Typical of white soil constituents like diatomaceous earth, pure quartz sand, or significant carbonate/gypsum accumulations.",
    "properties": ["High Purity Light Minerals", "Low Organic Matter", "May Indicate Leached Horizons (Albic)", "Arid/Semi-Arid Deposits Common"]
  },
  "N 8.5/": {
    "name": "White",
    "hex": "#DADADA",
    "description": "N 8.5/ - White (Neutral). High value, neutral, slightly darker than N 9/. Still represents very light-colored materials with minimal pigmentation.",
    "properties": ["Light-Colored Minerals", "Very Low Pigmentation", "Low Organic Content", "May be finely powdered rock like limestone"]
  },
  "N 8/": {
    "name": "White",
    "hex": "#CDCDCD",
    "description": "N 8/ - White (Neutral). Moderately high value, neutral. A light gray rather than pure white. Can be from light-colored parent rock or bleached materials.",
    "properties": ["Light Gray Appearance", "Low Organic Matter", "Bleached Minerals or Light Parent Rock"]
  },
  "7.5YR 9.5/1": {
    "name": "White",
    "hex": "#F5F4F2",
    "description": "7.5YR 9.5/1 - White with a very faint pinkish cast. Extremely high value, very low chroma. Indicates nearly pure white material with a trace of reddish-yellow iron staining.",
    "properties": ["Trace Iron Staining (Reddish-Yellow)", "Extremely Light Material", "Very Low Organic Matter", "High Reflectance"]
  },
  "7.5YR 9/1": {
    "name": "Pinkish White",
    "hex": "#E8E7E5",
    "description": "7.5YR 9/1 - Pinkish White. High value, very low chroma. White with a faint but discernible pinkish-beige hue from slight iron oxide presence.",
    "properties": ["Slight Iron Oxide Presence (Pinkish-Beige)", "Low Organic Matter", "Calcareous or Siliceous Base"]
  },
  "7.5YR 8.5/1": {
    "name": "Pinkish White",
    "hex": "#DAD9D7",
    "description": "7.5YR 8.5/1 - Pinkish White. High value, very low chroma. Similar to 9/1 but slightly darker, still a very light material with a faint warm tint.",
    "properties": ["Faint Warm Tint (Iron)", "Light-Colored Mineral Base", "Low Organic Content"]
  },
  "7.5YR 8/1": {
    "name": "Pinkish Gray",
    "hex": "#CDCCCB",
    "description": "7.5YR 8/1 - Pinkish Gray. Moderately high value, very low chroma. A light gray with a noticeable pinkish or pale brownish hue.",
    "properties": ["Noticeable Pinkish/Brownish Hue", "Low Organic Matter", "Often from Weathered Granitic Rocks or Light Loess"]
  },
  "7.5YR 9.5/2": {
    "name": "Pinkish White",
    "hex": "#F5F0ED",
    "description": "7.5YR 9.5/2 - Pinkish White. Extremely high value, low chroma. Very pale pinkish-beige, slightly more chroma than /1. Trace iron oxides.",
    "properties": ["More Pronounced Trace Iron Oxides", "Extremely Light Material", "Very Low Organic Matter"]
  },
  "7.5YR 9/2": {
    "name": "Pinkish White",
    "hex": "#E8E0DB",
    "description": "7.5YR 9/2 - Pinkish White. High value, low chroma. A pale pinkish-beige or very light tan, indicating a slightly higher concentration of iron oxides than 9/1.",
    "properties": ["Higher Concentration of Iron Oxides (Pinkish-Beige)", "Low Organic Matter", "Well-Aerated Light Material"]
  },
  "7.5YR 8.5/2": {
    "name": "Pink",
    "hex": "#DAD2CD",
    "description": "7.5YR 8.5/2 - Pink. High value, low chroma. A pale but definite pink or very light brown with a reddish cast.",
    "properties": ["Definite Pink/Light Brown (Reddish Cast)", "Light-Colored Mineral Base with Iron", "Low Organic Content"]
  },
  "7.5YR 8/2": {
    "name": "Pink",
    "hex": "#CDC5C0",
    "description": "7.5YR 8/2 - Pink. Moderately high value, low chroma. A light brownish-pink or pale reddish-brown.",
    "properties": ["Light Brownish-Pink/Pale Reddish-Brown", "Low Organic Matter", "Common in some arid soils or weathered sediments"]
  },
  "10YR 9.5/1": {
    "name": "White",
    "hex": "#F5F5F3",
    "description": "10YR 9.5/1 - White with a very faint yellowish-beige cast. Extremely high value, very low chroma. Nearly pure white with a trace of yellowish iron staining.",
    "properties": ["Trace Iron Staining (Yellowish-Beige)", "Extremely Light Material", "Very Low Organic Matter"]
  },
  "10YR 9/1": {
    "name": "White",
    "hex": "#E8E8E6",
    "description": "10YR 9/1 - White with a faint yellowish-beige hue. High value, very low chroma. Slight yellowish iron oxide presence in white material.",
    "properties": ["Slight Iron Oxide Presence (Yellowish-Beige)", "Low Organic Matter", "Calcareous or Siliceous Base"]
  },
  "10YR 8.5/1": {
    "name": "Very Pale Brown",
    "hex": "#DADAD8",
    "description": "10YR 8.5/1 - Very Pale Brown (off-white). High value, very low chroma. Very light yellowish-beige with minimal pigmentation.",
    "properties": ["Minimal Pigmentation (Yellowish-Beige)", "Light-Colored Mineral Base", "Low Organic Content"]
  },
  "10YR 8/1": {
    "name": "Very Pale Brown",
    "hex": "#CDCDCC",
    "description": "10YR 8/1 - Very Pale Brown (light gray with yellowish tint). Moderately high value, very low chroma. Light gray with a noticeable yellowish or pale brownish hue.",
    "properties": ["Noticeable Yellowish/Pale Brownish Hue", "Low Organic Matter", "Often from light-colored loess or alluvium"]
  },
  "10YR 9.5/2": {
    "name": "Very Pale Brown",
    "hex": "#F5F1EE",
    "description": "10YR 9.5/2 - Very Pale Brown. Extremely high value, low chroma. Very pale yellowish-beige, slightly more chroma than /1. Trace yellowish iron oxides.",
    "properties": ["More Pronounced Trace Yellowish Iron Oxides", "Extremely Light Material", "Very Low Organic Matter"]
  },
  "10YR 9/2": {
    "name": "Very Pale Brown",
    "hex": "#E8E1DB",
    "description": "10YR 9/2 - Very Pale Brown. High value, low chroma. A pale yellowish-beige or very light tan, indicating a slightly higher concentration of yellowish iron oxides (goethite) than 9/1.",
    "properties": ["Higher Concentration of Goethite (Yellowish-Beige)", "Low Organic Matter", "Well-Aerated Light Material"]
  },
  "10YR 8.5/2": {
    "name": "Very Pale Brown",
    "hex": "#DAD3CE",
    "description": "10YR 8.5/2 - Very Pale Brown. High value, low chroma. A pale but definite yellowish-beige or very light tan.",
    "properties": ["Definite Yellowish-Beige/Light Tan", "Light-Colored Mineral Base with Goethite", "Low Organic Content"]
  },
  "10YR 8/2": {
    "name": "Very Pale Brown",
    "hex": "#CDC6C1",
    "description": "10YR 8/2 - Very Pale Brown. Moderately high value, low chroma. A light yellowish-tan or pale beige-brown.",
    "properties": ["Light Yellowish-Tan/Pale Beige-Brown", "Low Organic Matter", "Common in arid or semi-arid soils, or C horizons"]
  },
  "2.5Y 9.5/1": {
    "name": "White",
    "hex": "#F5F5F3",
    "description": "2.5Y 9.5/1 - White with a very faint pale yellow cast. Extremely high value, very low chroma. Nearly pure white with a trace of pale yellow mineral influence.",
    "properties": ["Trace Pale Yellow Mineral Influence", "Extremely Light Material", "Very Low Organic Matter"]
  },
  "2.5Y 9/1": {
    "name": "White",
    "hex": "#E8E8E6",
    "description": "2.5Y 9/1 - White with a faint pale yellow hue. High value, very low chroma. Slight pale yellow mineral presence in white material.",
    "properties": ["Slight Pale Yellow Mineral Presence", "Low Organic Matter", "Calcareous or Siliceous Base"]
  },
  "2.5Y 8.5/1": {
    "name": "White",
    "hex": "#DADAD8",
    "description": "2.5Y 8.5/1 - White (off-white with pale yellow tint). High value, very low chroma. Very light pale yellow with minimal pigmentation.",
    "properties": ["Minimal Pigmentation (Pale Yellow)", "Light-Colored Mineral Base", "Low Organic Content"]
  },
  "2.5Y 8/1": {
    "name": "Light Gray",
    "hex": "#CDCDCC",
    "description": "2.5Y 8/1 - Light Gray (with pale yellow tint). Moderately high value, very low chroma. Light gray with a noticeable pale yellow hue.",
    "properties": ["Noticeable Pale Yellow Hue", "Low Organic Matter", "Often from light-colored sedimentary rocks or alluvium"]
  },
  "2.5Y 9.5/2": {
    "name": "White",
    "hex": "#F5F1EF",
    "description": "2.5Y 9.5/2 - White with a pale yellow cast. Extremely high value, low chroma. Very pale yellow, slightly more chroma than /1. Trace pale yellow minerals.",
    "properties": ["More Pronounced Trace Pale Yellow Minerals", "Extremely Light Material", "Very Low Organic Matter"]
  },
  "2.5Y 9/2": {
    "name": "Pale Yellow",
    "hex": "#E8E1DC",
    "description": "2.5Y 9/2 - Pale Yellow. High value, low chroma. A pale light yellow, indicating a slightly higher concentration of yellowish minerals (e.g., goethite, jarosite traces) than 9/1.",
    "properties": ["Higher Concentration of Yellowish Minerals", "Low Organic Matter", "Well-Aerated Light Material"]
  },
  "2.5Y 8.5/2": {
    "name": "Pale Yellow",
    "hex": "#DAD3CE",
    "description": "2.5Y 8.5/2 - Pale Yellow. High value, low chroma. A pale but definite light yellow.",
    "properties": ["Definite Light Yellow", "Light-Colored Mineral Base with Yellowish Minerals", "Low Organic Content"]
  },
  "2.5Y 8/2": {
    "name": "Pale Yellow",
    "hex": "#CDC6C1",
    "description": "2.5Y 8/2 - Pale Yellow. Moderately high value, low chroma. A light yellowish color.",
    "properties": ["Light Yellowish Color", "Low Organic Matter", "Common in some loess or light-colored parent materials"]
  }
}

class MunsellClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
       
        # Verify the actual input shape from the model
        self.input_shape = self.model.input_shape[1:3]  # Gets (height, width)
        print(f"Model expects input shape: {self.input_shape}")
       
        self.class_names = list(MUNSELL_COLORS.keys())


    def preprocess_single_image_for_resnet50_prediction(self, image_path, config):
        """
        Loads and preprocesses a single image to be compatible with the trained
        ResNet50-based model.
        Assumes images were resized, then patched, then patches fed to ResNet50.
        """
        try:
            # --- 1. Load Image ---
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            print(f"Original image size: {original_size}")


            # --- 2. Initial Resize (to the size before patching, if applicable) ---
            initial_resize_h, initial_resize_w, _ = config.get('input_shape', (256, 256, 3)) # Default if not in config
            img_resized_for_patching = img.resize((initial_resize_w, initial_resize_h))
            print(f"Resized for patching to: {img_resized_for_patching.size}")
            img_array_for_patching = np.array(img_resized_for_patching, dtype=np.uint8)


            # --- 3. Extract Patch(es) ---
            patch_size_h = config['patch_size']
            patch_size_w = config['patch_size']
            patch_to_process = None # Initialize


            if img_array_for_patching.shape[0] < patch_size_h or img_array_for_patching.shape[1] < patch_size_w:
                print(f"Warning: Image resized for patching ({img_array_for_patching.shape[0]}x{img_array_for_patching.shape[1]}) "
                    f"is smaller than patch_size ({patch_size_h}x{patch_size_w}).")
                print("Attempting to resize the initially resized image directly to patch_size.")
                patch_to_process_pil = img_resized_for_patching.resize((patch_size_w, patch_size_h))
                patch_to_process = np.array(patch_to_process_pil, dtype=np.float32)
            else:
                if patchify and config.get('use_patching_for_prediction', True):
                    print(f"Extracting patches of size {patch_size_h}x{patch_size_w}...")
                    step = config.get('patch_step', patch_size_h // 2)
                    patches = patchify.patchify(img_array_for_patching, (patch_size_h, patch_size_w, 3), step=step)
                    center_y = patches.shape[0] // 2
                    center_x = patches.shape[1] // 2
                    extracted_patch_uint8 = patches[center_y, center_x, 0, :, :, :]
                    patch_to_process = extracted_patch_uint8.astype(np.float32)
                    print(f"Extracted central patch of size: {patch_to_process.shape}")
                else:
                    print(f"Taking a center crop of size {patch_size_h}x{patch_size_w}...")
                    left = (img_array_for_patching.shape[1] - patch_size_w) // 2
                    top = (img_array_for_patching.shape[0] - patch_size_h) // 2
                    right = left + patch_size_w
                    bottom = top + patch_size_h
                    center_cropped_patch_uint8 = img_array_for_patching[top:bottom, left:right, :]
                    patch_to_process = center_cropped_patch_uint8.astype(np.float32)
                    print(f"Center cropped patch size: {patch_to_process.shape}")


            # --- 4. Apply ResNet50-Specific Preprocessing ---
            preprocessed_patch = resnet50_preprocess_input(patch_to_process.copy()) # Use .copy() to avoid issues with read-only arrays
            print(f"Patch preprocessed for ResNet50. Shape: {preprocessed_patch.shape}, Min: {preprocessed_patch.min():.2f}, Max: {preprocessed_patch.max():.2f}")


            # --- 5. Add Batch Dimension ---
            batch_patch = np.expand_dims(preprocessed_patch, axis=0)
            print(f"Final batch shape for model: {batch_patch.shape}")


            return batch_patch


        except FileNotFoundError:
            print(f"❌ Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"❌ Error during preprocessing for ResNet50: {e}")
            import traceback
            traceback.print_exc()
            return None




    def predict_on_image_resnet50(self, image_path, config, class_names_list):
        """Loads a saved ResNet50-based model, preprocesses an image, makes a prediction,
        and returns the results as JSON."""
        try:
            # Load model
            # model = tf.keras.models.load_model(model_path, compile=False)
           
            # --- Preprocess Image ---
            processed_image_batch = self.preprocess_single_image_for_resnet50_prediction(image_path, config)
            # --- Make Prediction ---
            try:
                print("\nMaking prediction...")
                predictions_proba = self.model.predict(processed_image_batch)


                predicted_class_index = np.argmax(predictions_proba[0])
                predicted_class_probability = np.max(predictions_proba[0])
                predicted_class_name = "N/A"


                if 0 <= predicted_class_index < len(class_names_list):
                    predicted_class_name = class_names_list[predicted_class_index]
                    print(f"\n✅ Predicted Class: {predicted_class_name}")
                    print(f"   Confidence: {predicted_class_probability:.4f}")
                else:
                    print(f"⚠️ Error: Predicted class index ({predicted_class_index}) is out of bounds for class_names (len: {len(class_names_list)}).")
                    print(f"   This usually means the loaded class_names_list is incorrect or too short.")
                    print(f"   Raw probabilities (first 10): {predictions_proba[0][:10]}...")


                top_n = min(5, len(class_names_list) if len(class_names_list) > 0 else 1)
                top_n_indices = np.argsort(predictions_proba[0])[-top_n:][::-1]
                print(f"\nTop {top_n} predictions:")
                for i in top_n_indices:
                    if 0 <= i < len(class_names_list):
                        print(f"  - {class_names_list[i]}: {predictions_proba[0][i]:.4f}")
                    else:
                        print(f"  - Index {i} (out of bounds for class_names_list): {predictions_proba[0][i]:.4f}")


            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                import traceback
                traceback.print_exc()
           
            results = []
            for idx in top_n_indices:
                if 0 <= idx < len(class_names_list):
                    munsell_code = class_names_list[idx]
                    confidence = float(predictions_proba[0][idx])                    


                    # Try to get color info if available
                    color_data = MUNSELL_COLORS.get(munsell_code, {})
                   
                    results.append({
                        "munsell_code": munsell_code,
                        "color_name": color_data.get('name', f"Unknown Color {munsell_code}"),
                        "hex_color": color_data.get('hex', '#AAAAAA'),
                        "confidence": confidence,
                        "description": color_data.get('description', 'No description available for this color code.'),
                        "properties": color_data.get('properties', []),
                    })
           
            if not results:
                return {"error": "No valid predictions generated"}
               
            return {
                "predictions": results,
                "primary_prediction": results[0]
            }
           
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
   
    def preprocess_image(self, image):
        """Process image to match model's expected input"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize to model's expected dimensions
        image = image.resize(self.input_shape)
        img_array = np.array(image) / 255.0  # Normalize to [0,1]
        return img_array
   
    def predict(self, file_obj):
        try:
            print("DEBUG: Starting prediction...")
            image = Image.open(file_obj)
            print(f"DEBUG: Image opened successfully, format: {image.format}, mode: {image.mode}, size: {image.size}")
           
            img_array = self.preprocess_image(image)
            print(f"DEBUG: Image preprocessed, shape: {img_array.shape}")
           
            img_array = np.expand_dims(img_array, axis=0)
            print(f"DEBUG: Added batch dimension: {img_array.shape}")
           
            print("DEBUG: Running model prediction...")
            preds = self.model.predict(img_array)[0]
            print(f"DEBUG: Prediction complete, output shape: {preds.shape}")
            print(f"DEBUG: Top 3 prediction values: {sorted(preds, reverse=True)[:3]}")
           
            # Check if there's a mismatch
            # if preds.shape[0] != len(self.class_names):
            #     return {"error": f"Model output dimension ({preds.shape[0]}) doesn't match number of classes ({len(self.class_names)})"}
           
            top_indices = np.argsort(-preds)[:min(5, len(preds))]
            print(f"DEBUG: Top indices: {top_indices}")
            print(f"DEBUG: Class names available: {len(self.class_names)}")
           
            results = []
            for idx in top_indices:
                # Ensure index is valid
                if idx < len(self.class_names):
                    munsell_code = self.class_names[idx]
                    color_data = MUNSELL_COLORS.get(munsell_code, {})
                    results.append({
                        "munsell_code": munsell_code,
                        "color_name": color_data.get('name', 'Unknown'),
                        "hex_color": color_data.get('hex', '#FFFFFF'),
                        "confidence": float(preds[idx]),
                        "description": color_data.get('description', 'No description available'),
                        "properties": color_data.get('properties', []),
                    })
           
            # Check if results is empty
            if not results:
                return {"error": "No valid predictions generated"}
               
            return {
                "predictions": results,
                "primary_prediction": results[0]
            }
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
       


def temp(file):
    try:
        # Save the uploaded file temporarily
        temp_path = file.name
        print(f"DEBUG: Saving uploaded file to {temp_path}")


        # Call your predict method with the required parameters
        result = classifier.predict_on_image_resnet50(
            temp_path,
            config_for_resnet50_prediction,
            class_names_for_resnet50_prediction
        )
       
        # Return the results in the expected format
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


config_for_resnet50_prediction = {
    'input_shape': (150, 150, 3),
    'patch_size': 100,
    'patch_step': 50,
    'use_patching_for_prediction': True,
}


class_names_npy_path = "class_info.npy"


class_names_for_resnet50_prediction = []
num_classes_from_file_load = 0


loaded_data = np.load(class_names_npy_path, allow_pickle=True)
print(f"DEBUG: 0-D array: {loaded_data}")


# --- MODIFIED LOGIC TO HANDLE 0-D ARRAY CONTAINING A DICTIONARY ---
if loaded_data.ndim == 0 and isinstance(loaded_data.item(), dict):
    print(f"✓ Loaded a 0-D array containing a dictionary from '{class_names_npy_path}'.")
    data_dict = loaded_data.item()
    if 'class_names' in data_dict and isinstance(data_dict['class_names'], list):
        class_names_for_resnet50_prediction = [str(name) for name in data_dict['class_names']]
        print(f"  Extracted 'class_names' list with {len(class_names_for_resnet50_prediction)} items.")
    else:
        print(f"  ⚠️ Dictionary in .npy file does not contain a 'class_names' key with a list value.")
elif loaded_data.ndim == 1:
    print(f"✓ Loaded a 1-D array from '{class_names_npy_path}'.")
    class_names_for_resnet50_prediction = list(map(str, loaded_data))
else:
    print(f"⚠️ Loaded an array with unexpected dimensions (shape: {loaded_data.shape}) from '{class_names_npy_path}'. Expected 0-D with dict or 1-D array.")
# --- END OF MODIFIED LOGIC ---


if not class_names_for_resnet50_prediction:
    print(f"⚠️ Class names list is empty after processing '{class_names_npy_path}'.")
else:
    num_classes_from_file_load = len(class_names_for_resnet50_prediction)
    print(f"✓ Successfully processed {num_classes_from_file_load} class names.")

    

# ✅ Instantiate classifier
classifier = MunsellClassifier("munsell_classifier.keras")

# ✅ Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"status": "soil color classifier ready"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        result = temp(file)
        return result
    except Exception as e:
        return {"error": str(e)}

