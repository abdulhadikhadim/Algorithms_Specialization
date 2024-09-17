class Query:
    def __init__(self, query):
        self.type = query[0]
        self.number = int(query[1])
        if self.type == 'add':
            self.name = query[2]


def read_queries():
    n = int(input())
    return [Query(input().split()) for _ in range(n)]


def write_responses(result):
    print('\n'.join(result))


def process_queries(queries):
    result = []
    phone_book = {}

    for cur_query in queries:
        if cur_query.type == 'add':
            # Add or update the name for the given number
            phone_book[cur_query.number] = cur_query.name
            print(cur_query)
        elif cur_query.type == 'del':
            # Remove the number from the phone book if it exists
            if cur_query.number in phone_book:
                del phone_book[cur_query.number]
        else:  # cur_query.type == 'find'
            # Look up the number and return the associated name or 'not found'
            response = phone_book.get(cur_query.number, 'not found')
            result.append(response)

    return result


if __name__ == '__main__':
    write_responses(process_queries(read_queries()))
