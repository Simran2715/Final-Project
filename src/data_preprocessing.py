import pandas as pd
import numpy as np

n=5000
Student_ID=[i for i in range(1,n+1)]
Time_Spent_ON_App=np.random.uniform(0,10,size=n)
past_grades=np.random.randint(0,101,size=n)
attendence_rate=np.random.randint(50,101, size=n)
time_spent_on_quiz= np.random.uniform(0,5,size=n)
question_attemted=np.random.randint(0,50,size=n)
pass_fail = np.where(past_grades > 40, 'Pass', 'Fail')
noise = np.random.normal(0, 5, n)
future_score = (
    60 + (past_grades * 0.4)
    - (time_spent_on_quiz * 0.1)
    + noise
)

topic_difficulty = np.where(past_grades < 40, 'High',
                            np.where((past_grades >= 40) & (past_grades < 60), 'Medium', 'Low'))
data={
    'Student_ID' : Student_ID,
    'Time_Spent_On_App' : Time_Spent_ON_App,
    'Past_Grades' : past_grades,
    'attendance_rate' : attendence_rate,
    'time_spent_on_quiz' : time_spent_on_quiz,
    'question_attemted' : question_attemted,
    'pass_fail' : pass_fail,
    'topic_difficulty': topic_difficulty,
    'future_score': future_score
}
df=pd.DataFrame(data)
df.to_csv('Student_Performance.csv', index=False)

student_performance=pd.read_csv('Student_Performance.csv')

#generating data for dropout risk detector
num_samples=5000

np.random.seed(42) # for reproducibility

data = {
    'Student_ID' : [i for i in range(1,num_samples+1)],
    'inactivity_score': np.random.randint(0, 101, num_samples),
    'poor_performance_score': np.random.randint(0, 101, num_samples),
    'inconsistent_engagement_score': np.random.randint(0, 101, num_samples),
    'study_hours_per_week': np.random.randint(0, 41, num_samples),
    'attendance_rate': student_performance['attendance_rate'], # Assume minimum 50% attendance
    'previous_failures': np.random.randint(0, 6, num_samples)
}
df = pd.DataFrame(data)

    # Introduce some correlation for 'dropout'
    # Higher inactivity, poor performance, inconsistent engagement, more failures
    # and lower study hours/attendance are more likely to lead to dropout.
dropout_prob = (
    0.05 + # Base dropout probability
    df['inactivity_score'] * 0.002 +
    df['poor_performance_score'] * 0.003 +
    df['inconsistent_engagement_score'] * 0.0025 +
    df['previous_failures'] * 0.05 -
    df['study_hours_per_week'] * 0.005 -
    (100 - df['attendance_rate']) * 0.002
)
dropout_prob = np.clip(dropout_prob, 0.01, 0.99) # Clip probabilities to valid range

df['dropout'] = (np.random.rand(num_samples) < dropout_prob).astype(int)

print(f"Generated a synthetic dataset with {num_samples} samples.")
print("Dataset head:")
print(df.head())
print("\nDropout distribution:")
print(df['dropout'].value_counts())
print("-" * 50)
df.to_csv('dropout_risk.csv',index=False)

import random

#generating data for topic detection
# Define core sentences/phrases for each topic
topic_sentences = {
    "Mathematics": [
        "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. This is often written as $a^2 + b^2 = c^2$. It's a fundamental concept in geometry.",
        "Solving quadratic equations often involves factoring, completing the square, or using the quadratic formula. The formula, $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$, is particularly useful for complex cases.",
        "Calculus deals with rates of change and accumulation of quantities. Differentiation finds instantaneous rates of change, while integration finds the accumulation of quantities.",
        "Algebra involves symbols and the rules for manipulating these symbols. It's a fundamental branch of mathematics that provides a framework for solving equations and understanding relationships.",
        "The concept of infinity in mathematics represents a quantity without bound. It's crucial in calculus, especially when dealing with limits and series.",
        "To solve for $x$ in the equation $2x + 5 = 15$, first subtract $5$ from both sides, which gives $2x = 10$. Then, divide by $2$ to get $x = 5$.",
        "Understanding derivatives is key to grasping instantaneous rates of change. The derivative of $x^2$ is $2x$.",
        "Probability theory helps us quantify uncertainty. For example, the probability of rolling a 6 on a fair six-sided die is $1/6$.",
        "Geometry explores shapes, sizes, positions of figures, and properties of space. Euclidean geometry is a classical system of geometry.",
        "Statistics is the science of collecting, analyzing, interpreting, and presenting data. Mean, median, and mode are measures of central tendency."
    ],
    "Science": [
        "Photosynthesis is the process used by plants, algae and certain bacteria to convert light energy into chemical energy. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.",
        "Newton's laws of motion describe the relationship between a body and the forces acting upon it, and its motion in response to those forces. The first law is often called the law of inertia.",
        "The human circulatory system is responsible for transporting blood, nutrients, oxygen, carbon dioxide, and hormones throughout the body. It consists of the heart, blood vessels, and blood.",
        "Chemical reactions involve the rearrangement of atoms and molecules. Reactants are transformed into products, often with a change in energy, such as the release of heat in exothermic reactions.",
        "Genetics is the study of genes, genetic variation, and heredity in living organisms. DNA contains the instructions for building an organism.",
        "The theory of evolution by natural selection, proposed by Charles Darwin, explains how species change over time through adaptation to their environment.",
        "Cells are the basic structural, functional, and biological units of all known organisms. They are often called the 'building blocks of life'.",
        "Electromagnetism is a branch of physics that studies the interactions between electric current and magnetic fields. Maxwell's equations unify these phenomena.",
        "The periodic table organizes chemical elements by atomic number, electron configuration, and recurring chemical properties. It's a fundamental tool in chemistry.",
        "Ecosystems are communities of living organisms interacting with their physical environment. Food webs show the feeding relationships between organisms."
    ],
    "History": [
        "The causes of World War I were complex and included militarism, alliances, imperialism, and nationalism. The assassination of Archduke Franz Ferdinand was a key trigger, but underlying tensions had been building for years.",
        "The French Revolution, which began in 1789, led to the overthrow of the monarchy and the establishment of a republic. Its ideals of liberty, equality, and fraternity had a profound impact on European history.",
        "The ancient Roman Empire was vast and influential, known for its engineering, legal system, and military might. Its decline was a long and complex process, influenced by economic, social, and military factors.",
        "The American Civil War (1861-1865) was fought primarily over the issue of slavery and states' rights. Key figures included Abraham Lincoln and military generals like Ulysses S. Grant.",
        "The Cold War was a geopolitical rivalry between the United States and the Soviet Union and their respective allies, from the mid-1940s until the early 1990s. It was characterized by an arms race and proxy wars.",
        "The Industrial Revolution marked a period of significant technological advancement, beginning in the late 18th century, with the invention of the steam engine being a major catalyst.",
        "Ancient Egypt was a civilization of ancient Northeastern Africa, concentrated along the lower reaches of the Nile River. Its history is marked by dynasties of pharaohs and monumental architecture like the pyramids.",
        "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It saw a rebirth of classical art, literature, and philosophy.",
        "World War II, from 1939 to 1945, involved the vast majority of the world's countries forming two opposing military alliances: the Allies and the Axis. It was the deadliest conflict in human history.",
        "Colonialism is the policy or practice of acquiring full or partial political control over another country, occupying it with settlers, and exploiting it economically."
    ],
    "Literature": [
        "Shakespeare's Hamlet explores themes of revenge, morality, and corruption. The soliloquy 'To be, or not to be' is one of the most famous lines in English literature, contemplating life and death.",
        "Poetry often uses various literary devices like metaphor, simile, and personification to create vivid imagery and convey deeper meaning. Rhyme and rhythm also play significant roles.",
        "Literary analysis involves interpreting texts to understand their themes, characters, plot, and author's purpose. It often considers the historical and cultural context.",
        "Novels are extended prose narratives, typically exploring human experience through a sequence of events. They often feature complex characters and detailed settings.",
        "The epic poem 'The Odyssey' by Homer details Odysseus's long and perilous journey home after the Trojan War. It's a classic example of ancient Greek literature.",
        "Symbolism in literature uses objects or ideas to represent something else. For instance, a dove might symbolize peace.",
        "Literary themes are the underlying messages or main ideas that the author explores. Common themes include love, loss, courage, and redemption.",
        "Drama typically involves plays for theater, radio, or television. Elements include plot, character, theme, dialogue, and spectacle.",
        "Figurative language, like metaphors and similes, helps authors create more engaging and imaginative descriptions in their writing.",
        "A short story is a piece of prose fiction that can typically be read in a single sitting and focuses on a self-contained incident or series of linked incidents."
    ],
    "Geography": [
        "The theory of plate tectonics explains the movement of Earth's lithosphere. Continents drift over geological time due to convection currents in the mantle.",
        "The study of maps involves understanding projections, scales, and symbols. Topographic maps show elevation changes using contour lines.",
        "The Amazon rainforest is the largest tropical rainforest in the world. It plays a crucial role in global climate regulation and biodiversity.",
        "Climate zones are areas that have distinct climates, which are defined by their temperature and precipitation patterns. Examples include tropical, temperate, and polar zones.",
        "Rivers play a vital role in shaping landscapes, providing water for agriculture, and serving as transportation routes.",
        "Mountains are large natural elevations of the Earth's surface. They are formed through tectonic forces or volcanism.",
        "Deserts are arid regions characterized by extremely low precipitation. The Sahara Desert is the largest hot desert in the world.",
        "Urbanization is the process by which populations shift from rural to urban areas, leading to the growth of cities.",
        "Weather refers to the atmospheric conditions at a specific place and time, while climate describes long-term patterns of weather.",
        "Volcanoes are ruptures in the Earth's crust that allow hot lava, volcanic ash, and gases to escape from a magma chamber below the surface."
    ],
    "Art": [
        "Impressionism was an art movement characterized by small, thin, yet visible brush strokes, open composition, emphasis on accurate depiction of light in its changing qualities, ordinary subject matter, inclusion of movement as a crucial element of human perception and experience, and unusual visual angles.",
        "Renaissance art saw a revival of classical forms and a focus on humanism. Artists like Leonardo da Vinci and Michelangelo produced iconic works.",
        "Cubism, pioneered by Picasso and Braque, revolutionized painting by depicting subjects from multiple viewpoints simultaneously, often fracturing and reassembling them into abstract forms.",
        "Surrealism aimed to unleash the creative potential of the unconscious mind. Salvador Dalí is one of the most famous Surrealist painters.",
        "The use of color in painting can evoke different emotions and create depth. Primary colors are red, yellow, and blue.",
        "Sculpture is the art of making two- or three-dimensional representative or abstract forms. Materials include stone, metal, wood, and clay.",
        "Photography is the art, application, and practice of creating durable images by recording light or other electromagnetic radiation, either electronically by means of an image sensor, or chemically by means of a light-sensitive material such as photographic film.",
        "Abstract art does not attempt to represent external reality but seeks to achieve its effect using shapes, forms, colors, and textures.",
        "Gothic architecture, prevalent during the High and Late Middle Ages, is characterized by pointed arches, ribbed vaults, and flying buttresses.",
        "Pop Art emerged in the 1950s, challenging traditional fine art by including imagery from popular and mass culture, such as advertising and comic books."
    ],
    "Computer Science": [
        "Binary numbers are composed of only two digits: 0 and 1. They are fundamental in computer science because digital electronic circuits are built upon this system.",
        "Algorithms are a set of well-defined instructions for solving a problem or performing a computation. Efficiency of an algorithm is often measured by its time and space complexity.",
        "Object-Oriented Programming (OOP) is a paradigm based on the concept of 'objects', which can contain data and code. Key principles include encapsulation, inheritance, and polymorphism.",
        "Data structures like arrays, linked lists, and trees are fundamental ways to organize data for efficient access and modification in computer programs.",
        "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.",
        "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks.",
        "Artificial Intelligence (AI) is the intelligence of machines or software. It is the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        "Web development involves creating websites and web applications. Frontend development deals with the user interface, while backend development handles server-side logic.",
        "Databases are organized collections of data, generally stored and accessed electronically from a computer system. SQL is a common language for managing relational databases.",
        "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed. Supervised, unsupervised, and reinforcement learning are common types."
    ]
}

def generate_sentence(topic):
    """Generates a slightly varied sentence for a given topic."""
    base_sentence = random.choice(topic_sentences[topic])
    # Add minor variations or additional related phrases
    variations = {
        "Mathematics": ["This is a core concept.", "It's essential for advanced studies.", "Many real-world problems use this.", "It applies to various fields."],
        "Science": ["This biological process is vital.", "It demonstrates fundamental principles.", "Understanding this is key.", "This helps explain natural phenomena."],
        "History": ["This event shaped the world.", "Its consequences are still felt today.", "It's a turning point in human civilization.", "Key figures played crucial roles."],
        "Literature": ["This work is highly influential.", "Its themes resonate deeply.", "The author's style is remarkable.", "It provides insight into the human condition."],
        "Geography": ["This physical feature is significant.", "It impacts human settlement patterns.", "Understanding this is crucial for environmental studies.", "It's a key aspect of Earth's surface."],
        "Art": ["This movement challenged conventions.", "Its visual impact is profound.", "Artists experimented with new forms.", "It reflects cultural changes."],
        "Computer Science": ["This is a foundational concept.", "It enables powerful software.", "Efficiency is key in this area.", "It's central to modern technology."]
    }
    if topic in variations and random.random() > 0.3: # 70% chance to add variation
        return base_sentence + " " + random.choice(variations[topic])
    return base_sentence

# Generate 5000 entries
num_entries = 5000
generated_data = []
all_topics = list(topic_sentences.keys())

for _ in range(num_entries):
    chosen_topic = random.choice(all_topics)
    text_content = generate_sentence(chosen_topic)
    generated_data.append({"text": text_content, "label": chosen_topic})


num_enteries=5000
# To use this in Python code:
df = pd.DataFrame(generated_data)
texts = df['text'].values
labels = df['label'].values

df['Student_ID'] = [i for i in range(1,num_enteries+1)]

df.to_csv('topic_detection.csv',index=False)

num_students=5000

np.random.seed(42) # for reproducibility
data = []

    # Group 1: Fast Responders / High Achievers
for _ in range(num_students // 3):
    data.append({
        'reading_speed_wpm': np.random.normal(loc=250, scale=30),
        'video_watch_time_pct': np.random.uniform(0.4, 0.8),
        'quiz_response_time_sec': np.random.normal(loc=30, scale=10),
        'quiz_score_pct': np.random.normal(loc=90, scale=5),
        'time_on_platform_hrs': np.random.normal(loc=8, scale=2),
        'learning_style' : 'Fast Respondents'
    })
    
        # Group 2: Slow Learners / Methodical
for _ in range(num_students // 3):
    data.append({
        'reading_speed_wpm': np.random.normal(loc=120, scale=20),
        'video_watch_time_pct': np.random.uniform(0.6, 0.9),
        'quiz_response_time_sec': np.random.normal(loc=100, scale=20),
        'quiz_score_pct': np.random.normal(loc=75, scale=10),
        'time_on_platform_hrs': np.random.normal(loc=12, scale=3),
        'learning_style' : 'Slow Learners'
    })
        
        # Group 3: Visual Learners / Average Pace
for _ in range(num_students - 2 * (num_students // 3)):
    data.append({
        'reading_speed_wpm': np.random.normal(loc=180, scale=25),
        'video_watch_time_pct': np.random.normal(loc=0.9, scale=0.05),
        'quiz_response_time_sec': np.random.normal(loc=60, scale=15),
        'quiz_score_pct': np.random.normal(loc=80, scale=8),
        'time_on_platform_hrs': np.random.normal(loc=10, scale=2.5),
        'learning_style' : 'Visual Learners'
})
df = pd.DataFrame(data)
# df['Student_ID'] = [i for i in range(1,num_students+1)]

df.to_csv('student_response.csv',index=False)



dropout=pd.read_csv('dropout_risk.csv')

result = pd.merge(student_performance ,dropout , how="inner", on=["Student_ID", "attendance_rate"])

result.to_csv('Combined_data.csv')

result.info()

result.describe()

result.drop_duplicates(inplace=True)

result.isnull().sum()

