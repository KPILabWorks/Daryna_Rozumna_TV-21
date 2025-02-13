from collections import Counter

class ObjectTracker:
    _instances = Counter()
    _book_list = []  # Список для збереження назв та авторів книг
    
    def __init__(self):
        class_name = self.__class__.__name__
        ObjectTracker._instances[class_name] += 1
        
    @classmethod
    def get_instance_count(cls, class_name):
        """Повертає кількість створених екземплярів класу."""
        return cls._instances[class_name]
    
    @classmethod
    def get_all_counts(cls):
        """Повертає кількість екземплярів для всіх класів."""
        return dict(cls._instances)
    
    @classmethod
    def reset_counts(cls):
        """Скидає лічильник об'єктів."""
        cls._instances.clear()
        cls._book_list.clear()
    
    @classmethod
    def get_book_list(cls):
        """Повертає список всіх доданих книг."""
        return cls._book_list

class Book(ObjectTracker):
    def __init__(self, title, author):
        super().__init__()
        self.title = title
        self.author = author
        ObjectTracker._book_list.append(f"'{self.title}' by {self.author}")
    
    def __str__(self):
        return f"'{self.title}' by {self.author}"

if __name__ == "__main__":
    while True:
        print("\nМеню:")
        print("1. Додати книгу")
        print("2. Показати кількість створених книг")
        print("3. Показати всі лічильники")
        print("4. Вийти")
        
        choice = input("Виберіть опцію: ")
        
        if choice == "1":
            title = input("Введіть назву книги: ")
            author = input("Введіть автора книги: ")
            book = Book(title, author)
            print(f"Додано книгу: {book}")
        elif choice == "2":
            count = ObjectTracker.get_instance_count("Book")
            books = ObjectTracker.get_book_list()
            print(f"Кількість створених книг: {count}")
            if books:
                print("Список книг:")
                for book in books:
                    print(f" - {book}")
            else:
                print("Немає доданих книг.")
        elif choice == "3":
            print("Всі лічильники:", ObjectTracker.get_all_counts())
        elif choice == "4":
            print("Вихід...")
            break
        else:
            print("Невірний вибір, спробуйте ще раз.")
