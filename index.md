# Girls Day – KI-Bildklassifikation 🎉
**Willkommen** zum **Girls Day** am Lehrstuhl für Maschinelles Lernen, insbesondere Uncertainty Quantification der Universität Regensburg!

**Datum:** 23. April 2026 <br>
**Beginn:** 14:00 Uhr <br>
**Ort:** Lehrstuhl für Maschinelles Lernen, insbesondere Uncertainty Quantification, Universität Regensburg <br>

**Preisverleihung:** Der Preis geht an den **ersten Platz** im Leaderboard am Ende des Workshops.

<link rel="stylesheet" type="text/css" href="style.css">

## Leaderboard

<table>
  <thead>
    <tr>
      <th>Platz</th>
      <th>Pseudonym</th>
      <th>Genauigkeit</th>
    </tr>
  </thead>
  <tbody>
    {% assign sorted_leaderboard = site.data.leaderboard | sort: 'accuracy' | reverse %}
    {% for row in sorted_leaderboard %}
    <tr class="{% if forloop.first %}first-place{% endif %}">
      <td>{{ forloop.index }}</td>
      <td>{{ row.pseudonym }}</td>
      <td>{{ row.accuracy }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

## Wie kannst du teilnehmen?

**Merke dir dein Pseudonym**, um später im Leaderboard nachzuschauen, wie gut deine KI abgeschnitten hat!

## Wie funktioniert der Workshop?

Heute lernst du, wie eine KI Bilder erkennen kann – ganz ohne Programmierkenntnisse!

1. **Bilder aufnehmen** – Fotografiere vier verschiedene Gegenstände mit der Webcam.
2. **KI trainieren** – Lass die KI lernen, welcher Gegenstand welcher ist.
3. **Modell testen** – Schau, wie gut deine KI auf neuen Bildern abschneidet.
4. **Leaderboard** – Vergleiche deine Genauigkeit mit den anderen Teilnehmerinnen!

## Was kannst du gewinnen?

Die Teilnehmerin mit der besten Genauigkeit erhält einen kleinen Preis.

## Welche Vorkenntnisse brauche ich?

**Keine!** Der Workshop ist für alle gedacht, die neugierig auf KI sind – egal ob du dich vorher schon damit beschäftigt hast oder nicht.

## Was bedeutet Genauigkeit?

Wir haben ungefähr 1000 Test-Bilder selbst aufgenommen.
Deine Genauigkeit wird berechnet, indem die Anzahl der korrekt erkannten Bilder durch die Gesamtzahl der Test-Bilder geteilt wird.

Eine Genauigkeit von 50% bedeutet, dass deine KI 500 von 1000 Test-Bildern richtig zugeordnet hat.

## Was sagt meine Genauigkeit aus?

- **< 40%**: Gut gemacht! Du hast die Grundlagen der KI kennengelernt.
- **40% - 50%**: Sehr gut! Deine KI hat schon einiges gelernt.
- **50% - 65%**: Hervorragend! Du hast ein echtes Gespür für KI-Modelle.
- **> 65%**: Fantastisch! Du bist ein KI-Talent!

## Danke für deinen Besuch!

Wir freuen uns, dass du heute dabei bist, und hoffen, dass du Spaß hattest!
Falls dich das Thema KI und Daten weiter interessiert, schau gerne auf unserer [Website](https://www.uni-regensburg.de/informatik-data-science/studieren/studieninteressierte) vorbei.