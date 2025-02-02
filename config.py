state_abbreviation_mapping = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

disease_mapping_prediction = {  
    'J06': 'Upper Respiratory Infections',
    'J02': 'Acute Pharyngitis',
    "J12": "Parainfluenza virus pneumonia",
    'J00': 'Acute nasopharyngitis',
    'J10': 'Influenza',
    'J18': 'Pneumonia',
    'U07' :'Corona',
}

disease_mapping_detection = {
    'Z23': 'Immunization',
    'I10': 'Hypertension',
    'E78': 'Hyperlipidemia',
    'K21': 'GERD',
    'Z20': 'Viral Exposure',
    'M54': 'Low Back Pain',
    'E11': 'Diabetes',
    'E55': 'Vitamin D Deficiency',
    'J06': 'Respiratory Infection',
    'R10': 'Abdominal Pain',
    'Z03': 'Biological Agent Exposure',
    'Z11': 'Tuberculosis Screening',
    'D64': 'Anemia',
    'Z11': 'Viral Screening',
    'N39': 'UTI',
    'R05': 'Cough',
    'U07': 'COVID-19',
    'R06': 'Shortness of Breath',
    'E03': 'Hypothyroidism',
    'E78': 'Mixed Hyperlipidemia',
    'K64': 'Hemorrhoids',
    'R07': 'Chest Pain',
    'K59': 'Constipation',
    'I25': 'Heart Disease',
    'B35': 'Tinea Unguium',
    'J30': 'Allergy',
    'J02': 'Acute Pharyngitis',
    'J10': 'Influenza',
    'B54': 'Malaria',
    'B15': 'Hepatitis A',
    'B17': 'Acute Viral Hepatitis',
    'J18': 'Pneumonia',
    'R19' : 'Diarrhea'
   
}


disease_traits_mapping = {
    'J06': {  # Upper Respiratory Infections
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 1.4,                      # R₀ varies; common cold viruses range from 1.2 to 1.8
        'Recovery Period': 7,            # Typically recover within 7-10 days
    },
    'J02': {  # Acute Pharyngitis
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 1.8,                      # R₀ varies; streptococcal pharyngitis around 1.8
        'Recovery Period': 7,            # Recovery usually within 7 days
    },
    'J12': {  # Parainfluenza Virus Pneumonia
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 1.5,                      # R₀ for parainfluenza viruses ranges from 1.2 to 1.6
        'Recovery Period': 14,           # Recovery can take 1-2 weeks
    },
    'J15': {  # Pneumonia due to Mycoplasma pneumoniae
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 1.7,                      # R₀ estimated around 1.7
        'Recovery Period': 14,           # Recovery typically within 2 weeks
    },
    'J00': {  # Acute Nasopharyngitis (Common Cold)
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 2.5,                      # R₀ for rhinoviruses ranges from 2 to 3
        'Recovery Period': 7,            # Recovery usually within 7-10 days
    },
    'J10': {  # Influenza
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 1.5,                      # R₀ typically ranges from 1.3 to 1.8
        'Recovery Period': 7,            # Recovery usually within 7 days
    },
    'J18': {  # Pneumonia
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 1.2,                      # R₀ varies; bacterial pneumonia around 1.2
        'Recovery Period': 21,           # Recovery can take 2-3 weeks
    },
    'U07': {  # COVID-19
        'Transmission Mode': 'Droplet',  # Spread through respiratory droplets
        'R_0': 3.0,                      # R₀ estimated between 2.5 and 3.5
        'Recovery Period': 14,           # Recovery varies; mild cases ~2 weeks
    },
}