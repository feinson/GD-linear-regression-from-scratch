class Person:
    def __init__(self, age):
        self.age = age

    def update_age(self, new_age):
        self.age = new_age
        return self

harry = Person(3)
print(f"harry's age is {harry.age}")

sally = harry.update_age(4)

print(f"harry's age is {harry.age}")
print(f"sally's age is {sally.age}")