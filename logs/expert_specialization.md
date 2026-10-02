# What the experts learned

Routed 491,520 validation tokens through `checkpoints/moe.pt`.

## Consecutive tokens sharing their top-1 expert

Chance level with 16 experts is 1/16 = 6.25%.

| layer | consecutive tokens with the same top-1 expert |
|---|---|
| 0 | 8.8% |
| 1 | 24.0% |
| 2 | 23.2% |
| 3 | 27.8% |
| 4 | 29.4% |
| 5 | 29.1% |

## Layer 0: tokens each expert receives most consistently

For every token seen at least 200 times, the share of its occurrences routed to the expert (each token goes to 2 experts, so an even spread would be 12.5%).

| expert | load | top tokens (share of that token's occurrences) |
|---|---|---|
| 0 | 6.2% | `They` 99%, `But` 99%, ` explore` 99%, `"` 98%, `Once` 96%, `You` 94%, ` around` 89%, ` keep` 89% |
| 1 | 6.2% | ` is` 100%, ` to` 100%, ` home` 99%, ` outside` 99%, ` did` 97%, ` back` 95%, ` at` 94%, ` out` 91% |
| 2 | 6.2% | ` over` 99%, ` "` 99%, ` in` 97%, ` with` 93%, `s` 92%, ` â` 89%, ` or` 88%, ` and` 86% |
| 3 | 6.3% | ` lots` 100%, `ĠĊ` 100%, ` toys` 100%, ` together` 97%, ` an` 95%, ` some` 89%, ` its` 85%, ` his` 84% |
| 4 | 6.3% | ` excited` 98%, ` careful` 95%, `When` 95%, ` named` 94%, ` sad` 93%, `One` 87%, ` One` 86%, ` proud` 84% |
| 5 | 6.0% | ` slide` 100%, ` She` 99%, ` away` 99%, ` He` 97%, ` gave` 95%, ` They` 93%, ` best` 92%, ` put` 89% |
| 6 | 6.3% | ` food` 100%, ` garden` 100%, ` room` 100%, ` house` 100%, ` tree` 100%, `,"` 99%, ` things` 99%, `,` 99% |
| 7 | 6.3% | `!` 98%, `?` 97%, ` down` 96%, ` up` 93%, ` shiny` 83%, ` told` 81%, ` noise` 70%, ` say` 70% |
| 8 | 6.3% | ` because` 100%, `€™` 100%, ` for` 97%, `.` 96%, ` cat` 93%, ` too` 87%, ` food` 81%, ` inside` 77% |
| 9 | 6.1% | ` lot` 100%, ` toys` 100%, ` until` 100%, ` hard` 100%, ` other` 99%, ` again` 98%, ` before` 98%, ` new` 97% |
| 10 | 6.2% | `<\|endoftext\|>` 100%, ` with` 99%, ` big` 96%, ` and` 90%, ` hard` 90%, ` of` 86%, ` from` 81%, ` who` 75% |
| 11 | 6.2% | `'m` 100%, `I` 98%, ` wanted` 96%, ` saw` 95%, ` should` 95%, ` So` 95%, `?"` 94%, ` knew` 93% |
| 12 | 6.3% | ` hugged` 100%, ` car` 100%, ` played` 98%, ` laughed` 97%, ` walked` 97%, ` him` 96%, ` picked` 95%, ` family` 94% |
| 13 | 6.2% | `Mom` 100%, `I` 100%, `Yes` 100%, `But` 99%, `The` 99%, `You` 98%, ` mom` 97%, ` but` 97% |
| 14 | 6.4% | ` on` 100%, ` ball` 100%, ` upon` 95%, ` then` 85%, ` there` 84%, ` when` 82%, ` saw` 77%, ` Tim` 76% |
| 15 | 6.3% | ` sun` 100%, ` cat` 100%, ` ball` 100%, ` tree` 100%, ` It` 100%, ` little` 100%, ` bird` 100%, ` dog` 100% |

## Layer 3: tokens each expert receives most consistently

For every token seen at least 200 times, the share of its occurrences routed to the expert (each token goes to 2 experts, so an even spread would be 12.5%).

| expert | load | top tokens (share of that token's occurrences) |
|---|---|---|
| 0 | 5.6% | ` both` 76%, ` When` 71%, ` two` 71%, ` We` 68%, ` They` 65%, ` always` 63%, ` many` 60%, ` when` 56% |
| 1 | 6.4% | ` upon` 99%, `€™` 89%, ` each` 87%, ` every` 81%, ` One` 77%, ` time` 75%, ` little` 72%, ` there` 68% |
| 2 | 6.3% | ` dad` 69%, ` mommy` 59%, ` gave` 58%, ` walked` 57%, ` help` 53%, ` took` 52%, ` came` 52%, ` went` 50% |
| 3 | 6.0% | ` in` 74%, ` say` 64%, ` says` 58%, ` its` 56%, ` on` 56%, ` sun` 52%, ` We` 52%, `It` 49% |
| 4 | 6.3% | ` decided` 97%, ` thanked` 94%, ` told` 83%, ` felt` 74%, ` heard` 73%, ` feel` 68%, ` some` 67%, ` there` 60% |
| 5 | 5.9% | ` loved` 87%, ` tried` 83%, ` liked` 80%, ` wanted` 78%, ` picked` 72%, ` eat` 71%, `"` 70%, ` try` 69% |
| 6 | 6.1% | `Yes` 97%, `It` 81%, ` It` 65%, ` knew` 58%, ` thought` 57%, ` things` 57%, ` box` 54%, ` it` 53% |
| 7 | 6.2% | ` upon` 58%, `'m` 53%, `<\|endoftext\|>` 52%, ` being` 48%, ` a` 48%, ` named` 46%, ` didn` 43%, ` are` 42% |
| 8 | 6.3% | ` explore` 88%, `<\|endoftext\|>` 78%, ` or` 70%, ` hurt` 69%, ` before` 67%, ` know` 63%, ` heard` 60%, ` when` 57% |
| 9 | 6.8% | ` big` 93%, ` came` 88%, ` loud` 78%, ` red` 78%, ` small` 74%, ` toy` 72%, ` long` 65%, ` little` 62% |
| 10 | 6.7% | ` played` 100%, ` playing` 100%, ` play` 99%, ` learned` 94%, ` being` 82%, ` From` 76%, ` for` 68%, ` share` 64% |
| 11 | 6.5% | `The` 99%, ` The` 95%, ` the` 74%, ` Jack` 68%, ` go` 58%, ` went` 57%, ` Lucy` 56%, ` Timmy` 56% |
| 12 | 6.4% | ` safe` 93%, ` careful` 92%, ` better` 85%, ` again` 74%, ` end` 65%, ` food` 64%, ` now` 62%, ` lot` 59% |
| 13 | 6.1% | ` called` 100%, ` named` 99%, ` its` 79%, ` so` 75%, ` very` 70%, `'m` 69%, ` your` 67%, ` too` 66% |
| 14 | 6.4% | `One` 96%, ` One` 95%, ` So` 91%, ` lots` 86%, ` Then` 80%, ` also` 78%, `â` 78%, ` even` 75% |
| 15 | 6.0% | ` kind` 97%, ` ground` 92%, ` park` 89%, ` sorry` 88%, ` nice` 87%, ` slide` 83%, ` garden` 81%, ` great` 79% |

## Layer 5: tokens each expert receives most consistently

For every token seen at least 200 times, the share of its occurrences routed to the expert (each token goes to 2 experts, so an even spread would be 12.5%).

| expert | load | top tokens (share of that token's occurrences) |
|---|---|---|
| 0 | 6.1% | ` little` 78%, ` â` 70%, `€™` 68%, ` Her` 61%, ` its` 54%, ` His` 53%, `The` 50%, ` one` 44% |
| 1 | 6.8% | ` loved` 88%, ` lots` 75%, `ĠĊ` 68%, ` end` 67%, ` every` 57%, ` all` 55%, ` learned` 54%, ` great` 49% |
| 2 | 5.9% | ` â` 74%, `,"` 62%, ` liked` 50%, `<\|endoftext\|>` 50%, ` should` 48%, ` their` 46%, ` loved` 46%, ` played` 45% |
| 3 | 6.4% | `When` 100%, ` So` 98%, ` thanked` 98%, ` When` 97%, ` told` 96%, ` Then` 89%, ` until` 84%, ` gave` 79% |
| 4 | 5.9% | `One` 74%, ` excited` 74%, ` hard` 69%, ` proud` 67%, `You` 63%, ` angry` 63%, ` every` 61%, `Yes` 59% |
| 5 | 6.4% | ` from` 73%, ` The` 65%, ` wanted` 62%, ` into` 61%, ` saw` 60%, ` flew` 59%, `The` 57%, ` ran` 56% |
| 6 | 5.9% | `â` 65%, ` garden` 55%, ` laughed` 50%, ` room` 50%, ` park` 44%, ` Then` 43%, ` box` 42%, ` So` 40% |
| 7 | 5.9% | ` go` 73%, `Yes` 70%, ` me` 65%, ` careful` 63%, ` hurt` 61%, ` way` 57%, `Once` 55%, ` do` 51% |
| 8 | 6.5% | ` safe` 84%, ` keep` 82%, ` away` 81%, ` happy` 78%, ` home` 77%, ` back` 76%, ` better` 76%, ` From` 75% |
| 9 | 6.5% | ` flew` 90%, ` could` 70%, ` put` 66%, ` tried` 56%, ` started` 55%, ` will` 54%, ` would` 54%, ` water` 53% |
| 10 | 6.4% | ` there` 91%, ` explore` 87%, ` It` 86%, `It` 77%, ` You` 77%, ` eat` 71%, ` around` 68%, ` lived` 61% |
| 11 | 6.6% | ` named` 98%, `<\|endoftext\|>` 92%, ` park` 75%, ` friends` 69%, ` room` 68%, ` ground` 66%, ` garden` 65%, ` noise` 65% |
| 12 | 6.0% | ` an` 94%, ` a` 90%, ` new` 86%, ` big` 70%, ` great` 64%, ` One` 62%, ` old` 60%, ` special` 58% |
| 13 | 6.1% | ` feel` 93%, ` feeling` 92%, ` being` 91%, ` got` 91%, ` felt` 91%, `'m` 86%, ` get` 81%, ` very` 69% |
| 14 | 6.1% | ` upon` 82%, ` their` 58%, ` Her` 54%, ` His` 53%, `€™` 48%, ` his` 44%, ` your` 44%, `.` 43% |
| 15 | 6.3% | ` don` 95%, ` say` 82%, ` didn` 80%, ` said` 74%, ` time` 73%, ` loud` 73%, ` thought` 65%, ` says` 62% |
