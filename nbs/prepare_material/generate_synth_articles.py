import threading
from openai import OpenAI
import json
import dotenv
import time
from tqdm import tqdm

dotenv.load_dotenv()

client = OpenAI()
# subjects = [
#     'Cryptography',
#     'Quantum Computing', 'Blockchain', 'Cybersecurity',
#     'E-commerce', 'Society',
#     'Urban Planning', 'Social Media', 'Gaming', 'Virtual Reality',
#     'Space Exploration', 'Marine Biology', 'Renewable Energy', 'Public Health', 'Epidemiology',
#     'Psychology', 'Linguistics', 'Philosophy', 'Political Science', 'Economics',
#     'Sociology', 'Anthropology', 'History', 'Geography', 'Literature',
#     'Theater', 'Film Studies',
#     'Computer Vision', 'Speech Recognition', 'Human-Computer Interaction', 'Cyber-Physical Systems',
#     'Internet of Things', 'Wearable Technology', 'Telecommunications', 'Autonomous Vehicles', 'Smart Cities',
#     'Digital Marketing', 'Customer Service', 'Supply Chain Management', 'Human Resources', 'Real Estate',
#     'Hospitality', 'Travel and Tourism', 'Retail', 'Fashion', 'Fitness and Wellness',
#     'Mental Health', 'Nutrition', 'Gerontology', 'Pediatrics', 'Oncology',
#     'Cardiology', 'Neurology', 'Dermatology', 'Veterinary Medicine', 'Pharmacology',
#     'Biotechnology', 'Material Science', 'Chemical Engineering', 'Civil Engineering', 'Electrical Engineering',
#     'Mechanical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Petroleum Engineering', 'Mining',
#     'Forestry', 'Fisheries', 'Wildlife Conservation', 'Archaeology', 'Astrobiology',
#     'Climatology', 'Oceanography', 'Seismology', 'Volcanology', 'Geomatics',
#     'Horticulture', 'Entomology', 'Mycology', 'Hydrology', 'Toxicology',
#     'Endocrinology', 'Immunology', 'Microbiology', 'Virology', 'Structural Biology',
#     'Cognitive Science', 'Behavioral Economics', 'Demography', 'Cultural Studies',
#     'Algorithmic Trading', 'Drone Technology', 'Precision Agriculture',
#     'Language Translation', 'Facial Recognition Technology',
# ]

subjects = ['Aerospace Engineering', 'Agriculture', 'Algorithmic Trading',
       'Anthropology', 'Archaeology', 'Art', 'Astrobiology',
       'Astrophysics', 'Autonomous Vehicles', 'Behavioral Economics',
       'Bioinformatics', 'Biotechnology', 'Cardiology',
       'Chemical Engineering', 'Civil Engineering', 'Climate Change',
       'Climatology', 'Cognitive Science', 'Cryptography',
       'Cultural Studies', 'Customer Service', 'Cyber-Physical Systems',
       'Cybersecurity', 'Demography', 'Dermatology', 'Digital Marketing',
       'Drone Technology', 'E-commerce', 'Economics', 'Education',
       'Electrical Engineering', 'Endocrinology', 'Entomology',
       'Epidemiology', 'Fashion', 'Finance', 'Fisheries',
       'Fitness and Wellness', 'Food', 'Forestry', 'Gaming',
       'Genetic Engineering', 'Geography', 'Geomatics', 'Gerontology',
       'Healthcare', 'Horticulture', 'Hospitality', 'Human Resources',
       'Human-Computer Interaction', 'Immunology',
       'International Relations', 'Internet of Things',
       'Language Translation', 'Law', 'Linguistics', 'Logistics',
       'Marine Biology', 'Material Science', 'Mechanical Engineering',
       'Mental Health', 'Microbiology', 'Mining', 'Music', 'Mycology',
       'Nanotechnology', 'Neurology', 'Neuroscience',
       'Nuclear Engineering', 'Nutrition', 'Oceanography', 'Oncology',
       'Pediatrics', 'Petroleum Engineering', 'Pharmacology',
       'Philosophy', 'Political Science', 'Precision Agriculture',
       'Psychology', 'Public Health', 'Quantum Computing',
       'Renewable Energy', 'Retail', 'Robotics', 'Seismology',
       'Smart Cities', 'Society', 'Sociology', 'Space Exploration',
       'Structural Biology', 'Supply Chain Management',
       'Telecommunications', 'Theater', 'Toxicology', 'Transportation',
       'Travel and Tourism', 'Urban Planning', 'Virology',
       'Virtual Reality', 'Volcanology', 'Wearable Technology',
       'Wildlife Conservation']

def generate_prompt(subject, sentiment):
    sentiment_text = sentiment.lower()
    if sentiment_text == 'neutral':
        sentiment_text += ' (neither positive nor negative)'
    
    if sentiment_text == 'negative':
        sentiment_text += ' (negative, criticizing, or pessimistic)'
    
    return f'Write a short news article about "AI in {subject}" from a {sentiment_text} perspective.'
'The article should match in style articles found in MIT Tech Review, The Conversation, or AP News.'
'Make up a title for the article in the very first line, and follow it directly with a newline character.'
'The body should not exceed 512 tokens (including the title).'

def generate_prompts():
    for subject in subjects:
        for sentiment in ['positive', 'neutral', 'negative']:
            for i in range(10):
                prompt = generate_prompt(subject, sentiment)

                yield subject, sentiment, prompt

def generate_article(subject, sentiment, prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    text = response.choices[0].message.content
    title, body = text.split('\n', 1)

    global articles
    articles.append({
        'title': title,
        'body': body,
        'subject': subject,
        'sentiment': sentiment
    })

# Spawn a thread for every item in generate_prompts() and
# make the threads store the result in a global list
articles = []
threads = []

triples = list(generate_prompts())
for subject, sentiment, prompt in tqdm(triples, desc='Generating articles'):
    time.sleep(0.5)
    thread = threading.Thread(target=generate_article, args=(subject, sentiment, prompt))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

print(articles)
print(len(articles), len(subjects) * 3 * 10)

with open('articles.json', 'r') as fp:
    stored_articles = json.load(fp)

with open('articles_threaded.json', 'w') as fp:
    json.dump(stored_articles + articles, fp, indent=2)
