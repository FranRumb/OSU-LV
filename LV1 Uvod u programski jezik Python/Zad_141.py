def total_euro(radni_sati, satnica):
    return radni_sati*satnica


radniSati = input('Radni sati: ')
satnica = input('eura/h: ')
print('Ukupno: ', total_euro(int(radniSati), float(satnica)), 'eura')

