from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-8b-8192'  # Adjust based on available model
os.environ["OPENAI_API_KEY"] ='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'


email = "Hey, your neighbor John here, your house seems to be on fire. this is not a joke."
is_verbose = True

classifier = Agent(
    role = "email classifier",
    goal = "accurately classify emails based on their importance. give every email one of these ratings: important, casual, or spam",
    backstory = "You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be afraid to give emails bad rating if they are not important. Your job is to help the user manage their inbox.",
    verbose = is_verbose, 
    allow_delegation = False, 
    
)

responder = Agent( 
    role = "email responder",
    goal = "Based on the importance of the email, write a concise and simple response. If the email is rated 'important' write a formal response, if the email is rated 'casual' write a casual response, and if the email is rated 'spam' ignore the email. no matter what, be very concise",
    backstory = "You are an AI assistant whose only job is to write short responses to emails based on their importance. The importance will be provided to you by the 'classifier' agent.",
    verbose = is_verbose, 
    allow_delegation = False, 
    
)

classify_email = Task(
    description = f"Classify the following email: '{email}'",
    agent = classifier,
    expected_output = "One of these three options: 'important', 'casual', 'spam'",
)

respond_to_email = Task(
    description = f"Respond to the email: '{email}'",
    agent = responder,
    expected_output = "A short response to the email",
)

crew = Crew(
    agents = [classifier, responder],
    tasks = [classify_email, respond_to_email],
    verbose = 2,
    process = Process.sequential,
)

output = crew.kickoff()
print(output)