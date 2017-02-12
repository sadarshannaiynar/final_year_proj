import wikipedia
import rake

comp = wikipedia.page("Computer")
text = comp.content.replace("=", "").split("See also")[0].strip()
rake_object = rake.Rake("SmartStoplist.txt", 7, 8)
keywords = rake_object.run(text)
for keyword in keywords:
    if keyword[1] < 11 and keyword[1] > 6:
        print("Keyword: ", keyword[0], " Score: ", keyword[1])
