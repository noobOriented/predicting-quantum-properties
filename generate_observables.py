import itertools

import typer


app = typer.Typer()


@app.command()
def main(system_size: int):
    print(system_size)

    for i, j in itertools.product(range(system_size - 1), range(system_size - 1)):
        if j in (i - 1, i, i + 1):
            continue
        print(f'4 Y {i} Y {i + 1} X {j} X {j + 1}')

    for i, j in itertools.product(range(system_size - 1), range(system_size)):
        if j in (i, i + 1):
            continue

        for j2 in range(system_size):
            if j2 in (i, i + 1, j):
                continue
            print(f'4 X {i} X {i + 1} Z {j} Z {j2}')

    for i, j in itertools.product(range(system_size - 1), range(system_size)):
        if j in (i, i + 1):
            continue
        print(f'3 X {i} X {i + 1} Z {j}')


if __name__ == '__main__':
    app()
