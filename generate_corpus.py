import json
import random
from datetime import datetime, timedelta

def generate_chat_corpus(output_path="chat_corpus.json", target_count=4200):
    random.seed(42)  # Deterministic seed for reproducible evaluation ground truth IDs
    
    senders = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Kabir", "Neha"]
    
    start_date = datetime(2026, 3, 1, 8, 0, 0)
    end_date = datetime(2026, 8, 31, 23, 30, 0)
    total_seconds = int((end_date - start_date).total_seconds())
    
    # Pre-defined decision threads with exact message targets
    # We will insert these threads at specific percentages of the 6-month timeline
    decision_threads = [
        # Thread 1: Priya's Birthday (April 18-22, ~28% through timeframe)
        {
            "at_pct": 0.28,
            "topic": "priya_birthday",
            "messages": [
                ("Sneha", "Guys Priya ka birthday aane wala hai next week on April 22! Kya plan hai?"),
                ("Rohan", "Sahi yaad dilaya! We need to buy a nice surprise gift for her."),
                ("Aarav", "Usne last week mention kiya tha ki she really wants new wireless earphones."),
                ("Ananya", "Yes! Apple AirPods Pro 2nd Gen gift kar rahe hai Priya ko."), # TARGET: Priya birthday gift
                ("Vikram", "AirPods mast option hai. Total kitna kharcha aayega?"),
                ("Rohan", "Sab log suno, per person 1200 rupees GPay kar do mujhe fast."), # TARGET: Contribution amount
                ("Kabir", "Cool, phone number/UPI ID bata de jahan send karna hai."),
                ("Rohan", "GPay number is 9876543210 and UPI ID is rohan@upi, send screenshot after paying."), # TARGET: Payment details
                ("Neha", "Done, maine send kar diya rohan@upi pe! What about party dinner?"),
                ("Aarav", "Dinner ke liye restaurant reserve karna hai Saket side."),
                ("Sneha", "The Spice Route Saket at 8 PM for birthday dinner table booked under Rohan!"), # TARGET: Restaurant booking
                ("Priya", "Hey guys, group pe shor kyu chal raha hai? 🙈"),
                ("Vikram", "Kuch nahi Priya, just regular office ka rona dhona 😜"),
                ("Ananya", "Haha exact, party ka koi plan nahi hai don't worry 😂"),
            ]
        },
        # Thread 2: Flat Hunting (Late April / Early May, ~36% through timeframe)
        {
            "at_pct": 0.36,
            "topic": "flat_hunting",
            "messages": [
                ("Kabir", "Flat hunt update: Aaj broker ke saath 3 properties dekhi Gurgaon mein."),
                ("Vikram", "Kaunsi location sahi lagi? Metro close honi chahiye."),
                ("Kabir", "Sector 43 Gurgaon 3BHK flat finalized after negotiation."), # TARGET: Flat location
                ("Aarav", "Awesome! Monthly rent kitna decide hua final?"),
                ("Kabir", "Total rent 45000 is 15000 per person monthly excluding maintenance."), # TARGET: Flat rent
                ("Vikram", "15k per head is well within our budget. Deposit kitna maang raha landlord?"),
                ("Kabir", "2 months rent deposit final hai and half month brokerage."), # TARGET: Security deposit
                ("Sneha", "Possession aur shift karne ki date kab ki mili?"),
                ("Kabir", "Possession date is May 1st 2026, shift kar sakte hai us din."), # TARGET: Move-in date
                ("Neha", "Great job Kabir! Party banti hai flat milne pe 🎉"),
            ]
        },
        # Thread 3: Manali Trip (Early June, ~56% through timeframe)
        {
            "at_pct": 0.56,
            "topic": "manali_trip",
            "messages": [
                ("Aarav", "Summer heat is unbearable in Delhi NCR. Vacation trip plan kare?"),
                ("Priya", "Mountains chalte hai guys, Goa mein bohot garmi aur humidity hogi abhi."),
                ("Rohan", "Manali trip final karte hai for mid June!"),
                ("Ananya", "Dates final batao so that I can request leave at office."),
                ("Aarav", "June 12 to June 16 2026 final dates hain for Manali holiday."), # TARGET: Manali dates
                ("Sneha", "Hotel ya resort kaunsa book kar rahe hai Old Manali mein?"),
                ("Rohan", "Snow Peak Resort Old Manali book ho gaya with mountain view balconies."), # TARGET: Hotel booking
                ("Vikram", "Per head budget breakdown batao including travel and stay."),
                ("Aarav", "Sabka budget max 8500 per head final hua hai everything included."), # TARGET: Manali budget
                ("Kabir", "Transport mode kya fix hua? Train ya bus?"),
                ("Rohan", "Volvo bus departure 10 PM from ISBT Delhi on June 11 night."), # TARGET: Manali transport
                ("Neha", "Super excited! Packing list aur jacket ready rakho sab."),
            ]
        },
        # Thread 4: Goa Trip Cancellation (July, ~75% through timeframe)
        {
            "at_pct": 0.75,
            "topic": "goa_cancellation",
            "messages": [
                ("Priya", "Guys July end mein weekend trip to Goa ka kya bana?"),
                ("Vikram", "News dekho, heavy rainfall alerts and coastal warnings high hai Goa mein."),
                ("Sneha", "Monsoon mein Goa trip cancel ho gayi flooding hazard ki wajah se."), # TARGET: Goa cancellation
                ("Aarav", "Sad, but safety first. Red alert pe travel karna risk hai."),
                ("Rohan", "Flight tickets refund ho jayengi fortunately!"),
            ]
        },
        # Thread 5: IPL Final Screening (Late May, ~48% through timeframe)
        {
            "at_pct": 0.48,
            "topic": "ipl_screening",
            "messages": [
                ("Vikram", "IPL final match screening house party kahan kar rahe hai?"),
                ("Aarav", "Aarav ke ghar IPL final screening at 7:30 PM with big projector setup!"), # TARGET: IPL screening venue
                ("Neha", "Pizza order kaun karega? Dominoes discount coupon hai mere paas."),
                ("Rohan", "350 rupees per head for pizza and cold drinks pool kar lo."), # TARGET: IPL snacks cost
                ("Ananya", "Done, match ke waqt milte hai guys!"),
            ]
        }
    ]
    
    # Casual topics & Hinglish chat templates for filler messages
    hinglish_templates = [
        # Casual greetings & daily chatter
        "{sender}: Good morning guys! Aaj kisi ka office wfh hai kya?",
        "{sender}: Morning! Haan main aaj home se kaam kar raha hu.",
        "{sender}: Yaar aaj traffic bohot heavy hai DND flyover pe 🚗",
        "{sender}: Chai peene kaun chal raha hai niche Break room mein?",
        "{sender}: Aaj lunch mein kya laya hai koi?",
        "{sender}: Swiggy se Biryani order kare kya sab milke?",
        "{sender}: Haan bhai, count me in for biryani 🍲",
        "{sender}: Mujhe tight budget chal raha hai, main ghar ka tiffin laya hu.",
        "{sender}: Guys, project deadline extends ho gayi till Friday! Relief 😮‍💨",
        "{sender}: Sahi hai boss! Thoda chill scene hai phir aaj.",
        "{sender}: Kya chal raha hai sabka? Itna silence kyu hai group pe?",
        "{sender}: Sab office kaam mein busy hain yaar.",
        "{sender}: Haan, quarter end reviews chal rahe hain.",

        # Hinglish tech & work rants
        "{sender}: Yaar client requirement change kar raha hai har teen din mein 🤦‍♂️",
        "{sender}: IT job ka rona dhona kabhi khatam nahi hoga.",
        "{sender}: Zoom meeting mein 40 min se koi speaker muted bol raha tha 😂",
        "{sender}: Code production pe deploy kar diya, ab bas bhagwan bharose hai 🤞",
        "{sender}: Bug resolve ho gaya, simple typo tha config file mein.",
        "{sender}: Koffee break anytime soon?",
        "{sender}: Aaj sham ko CP mein drinks/coffee pe mile kya?",
        "{sender}: Aaj nahi bro, gym jana hai regular routine follow karna hai.",
        "{sender}: Gym bro 🏋️‍♂️ consistent rehna zaroori hai.",

        # Weekend & Movies & OTT
        "{sender}: Weekend pe naye movie release hui hai, koi dekhne chalega?",
        "{sender}: Reviews achhe nahi aaye hain IMDb pe, OTT release ka wait karo.",
        "{sender}: Netflix pe nayi crime documentary series dekhi kya kisine?",
        "{sender}: Binge watch kar daala poora season ek hi raat mein 🔥",
        "{sender}: Mast show hai yaar, background score is next level.",
        "{sender}: Saturday night party scene kahan hai?",
        "{sender}: Hauz Khas Village mein naya lounge open hua hai.",
        "{sender}: HKV mein parking ka bohot scene hota hai weekend pe.",

        # Typos, short replies, slang, reaction texts
        "{sender}: k",
        "{sender}: haan bilkul",
        "{sender}: nhi yaar",
        "{sender}: thik h tomw milte h",
        "{sender}: plzz share the link again",
        "{sender}: lol 🤣🤣",
        "{sender}: wbu?",
        "{sender}: full BT hai aaj office mein",
        "{sender}: kya jugaad banaya hai waah",
        "{sender}: mast hai veere",
        "{sender}: chill maro sab thik hoga",
        "{sender}: Forwarded: Good morning wish you a blessed day ahead with flowers 🌸☀️",
        "{sender}: Forwarded: Breaking News - Traffic advisory issued for Outer Ring Road due to rain.",
        "{sender}: Forwarded: Special discount offer 50% off on all sneakers code SNEAK50.",
        "{sender}: haha exact!",
        "{sender}: ok cool",
        "{sender}: hmmm",
        "{sender}: 💯",
        "{sender}: 👍",
        "{sender}: true that",
        "{sender}: bhai ye kya ho gaya 🤯",
        "{sender}: unexpected blooper!"
    ]

    all_messages = []
    
    # Calculate target indices for threads
    thread_insert_points = {}
    for dt in decision_threads:
        idx = int(dt["at_pct"] * target_count)
        thread_insert_points[idx] = dt["messages"]

    current_time = start_date
    time_increment_seconds = total_seconds // target_count

    msg_id = 1
    i = 0
    while msg_id <= target_count:
        # Check if a special thread starts at this index
        if i in thread_insert_points:
            thread_msgs = thread_insert_points[i]
            for sender, text in thread_msgs:
                # Add realistic time gaps between messages in a thread (10 seconds to 3 minutes)
                gap = random.randint(10, 180)
                current_time += timedelta(seconds=gap)
                
                all_messages.append({
                    "message_id": msg_id,
                    "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "sender": sender,
                    "text": text
                })
                msg_id += 1
            i += len(thread_msgs)
            continue

        # Regular ambient chatter message
        gap = random.randint(time_increment_seconds // 2, time_increment_seconds * 2)
        current_time += timedelta(seconds=gap)
        if current_time > end_date:
            current_time = end_date - timedelta(minutes=random.randint(1, 60))

        sender = random.choice(senders)
        template = random.choice(hinglish_templates)
        if "{sender}: " in template:
            text = template.replace("{sender}: ", "")
        else:
            text = template

        all_messages.append({
            "message_id": msg_id,
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sender": sender,
            "text": text
        })
        msg_id += 1
        i += 1

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Generated {len(all_messages)} WhatsApp messages to {output_path}")
    print(f"Date range: {all_messages[0]['timestamp']} to {all_messages[-1]['timestamp']}")
    return all_messages

if __name__ == "__main__":
    generate_chat_corpus()
