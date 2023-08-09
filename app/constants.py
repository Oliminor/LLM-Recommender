STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

CITIES = {'AL': ['Mobile',
  'Jasper',
  'Clanton',
  'Huntsville',
  'Birmingham',
  'Montgomery',
  'Prattville',
  'Enterprise',
  'Anderson',
  'Lincoln'],
 'AK': ['Healy', 'Juneau', 'Dutch Harbor'],
 'AZ': ['Glendale',
  'Tucson',
  'Eloy',
  'Phoenix',
  'Prescott Valley',
  'Mesa',
  'Payson',
  'Wellton'],
 'AR': ['Harrison',
  'Bull Shoals',
  'Yellville',
  'West Memphis',
  'Bryant',
  'Batesville',
  'Springdale',
  'Bella Vista',
  'Russellville'],
 'CA': ['Palm Springs',
  'Livermore',
  'Barstow',
  'Irvine',
  'San Diego',
  'Tracy',
  'Carlsbad',
  'Pasadena',
  'Marina',
  'San Jose',
  'Napa',
  'Marina Del Rey',
  'Rancho Mirage',
  'Sunnyvale',
  'San Francisco',
  'Colton',
  'Gardena',
  'San Bruno',
  'Morro Bay',
  'Joshua Tree',
  'Long Beach',
  'Anaheim',
  'Garberville',
  'Upland',
  'Eureka',
  'Inglewood',
  'Santa Barbara',
  'Redding',
  'Signal Hill',
  'Selma',
  'Lake Forest',
  'Garden Grove',
  'Mariposa',
  'San Clemente',
  'Studio City',
  'Rohnert Park',
  'Hesperia',
  'Victorville',
  'Coronado'],
 'CO': ['Silverton',
  'Castle Rock',
  'Denver',
  'Aspen',
  'Arvada',
  'Griante',
  'Cripple Creek',
  'Mosca',
  'Walden',
  'Colorado Springs',
  'Fort Collins',
  'Pueblo',
  'Grand Junction'],
 'CT': ['Bristol', 'Plainville', 'Southport', 'Hartford'],
 'DE': ['New Castle'],
 'FL': ['West Palm Beach',
  'Bonita Springs',
  'Orlando',
  'Palm Harbor',
  'Lake Worth',
  'Delray Beach',
  'Naples',
  'Jacksonville',
  'Deland',
  'Sarasota',
  'Bradenton',
  'Miami',
  'Englewood',
  'Fort Pierce'],
 'GA': ['Mableton',
  'Dublin',
  'Dahlonega',
  'Griffin',
  'Douglasville',
  'Macon',
  'Atlanta',
  'Norcross',
  'Warner Robins',
  'Calhoun',
  'Savannah',
  'Cave Spring',
  'Lithia Springs',
  'Madison',
  'Lagrange',
  'Silver Creek',
  'Lavonia',
  'Alpharetta',
  'Whigham'],
 'HI': ['Kapaa', 'Kihei', 'Princeville', 'Kailua Kona'],
 'ID': ['Gooding', 'Coeur D Alene', 'Boise', 'Idaho Falls'],
 'IL': ['Westmont',
  'Arcola',
  'Chicago',
  'Troy',
  'Rockford',
  'Northbrook',
  'Freeport',
  'Des Plaines',
  'Princeton',
  'Deerfield',
  'Galena',
  'Glenview',
  'Elk Grove Village'],
 'IN': ['Terre Haute',
  'Merrillville',
  'Bloomington',
  'Elkhart',
  'Indianapolis',
  'Fort Wayne',
  'Plainfield'],
 'IA': ['Humeston',
  'Coralville',
  'Ames',
  'Raymond',
  'Guernsey',
  'Boone',
  'Cedar Rapids'],
 'KS': ['Olathe', 'Manhattan', 'Ulysses'],
 'KY': ['Corbin',
  'Kuttawa',
  'Owensboro',
  'Hopkinsville',
  'Lexington',
  'Park City'],
 'LA': ['New Orleans',
  'Bossier City',
  'Opelousas',
  'Monroe',
  'Shreveport',
  'Slidell',
  'Sulphur'],
 'ME': ['Waterville', 'Ogunquit'],
 'MD': ['Columbia',
  'Pittsville',
  'Brandywine',
  'Glen Burnie',
  'Saint Leonard',
  'Annapolis',
  'Takoma Park'],
 'MA': ['Boston',
  'West Springfield',
  'Cambridge',
  'Brockton',
  'Auburn',
  'Fitchburg'],
 'MI': ['Holland',
  'Oscoda',
  'Bay City',
  'Prescott',
  'Utica',
  'Saginaw',
  'Southfield',
  'Grand Rapids',
  'Alma',
  'Saint Ignace',
  'Battle Creek',
  'Clinton Township',
  'Grayling',
  'Whitehall',
  'Big Rapids'],
 'MN': ['Harmony',
  'Detroit Lakes',
  'Shakopee',
  'Windom',
  'Worthington',
  'Minneapolis',
  'Fairmont',
  'Lake Elmo',
  'Burnsville',
  'Saint Cloud'],
 'MS': ['Batesville', 'Biloxi'],
 'MO': ['Osage Beach',
  'St. Robert',
  'Lamar',
  'Saint Mary',
  'Licking',
  'Saint Louis',
  'Joplin',
  'Sikeston',
  'Kansas City',
  'Springfield'],
 'MT': ['Whitefish',
  'Billings',
  'Kalispell',
  'Columbia Falls',
  'Miles City',
  'Big Sky'],
 'NE': ['Valentine', 'Burwell', 'Cozad', 'North Platte', 'Gretna'],
 'NV': ['Las Vegas', 'Incline Village', 'Laughlin'],
 'NH': ['Hampton', 'Twin Mountain', 'Northwood', 'Gorham'],
 'NJ': ['Clifton',
  'Springfield',
  'Mount Arlington',
  'Bordentown',
  'Ridgefield Park',
  'Carlstadt'],
 'NM': ['Albuquerque', 'Portales', 'Raton', 'Tatum'],
 'NY': ['Ithaca',
  'Staten Island',
  'Deansboro',
  'New York',
  'Hauppauge',
  'Geneva',
  'Cooperstown',
  'Newburgh',
  'East Syracuse',
  'Schenectady',
  'Albany',
  'Woodstock'],
 'NC': ['Lake Lure',
  'Forest City',
  'Cary',
  'Raleigh',
  'Charlotte',
  'Rocky Mount',
  'Southern Pines',
  'Lincolnton',
  'Burlington'],
 'ND': ['Devils Lake'],
 'OH': ['Alliance',
  'Stone Creek',
  'Dayton',
  'Bowling Green',
  'Columbus',
  'Portland',
  'Troy',
  'Springfield'],
 'OK': ['El Reno',
  'Oklahoma City',
  'Owasso',
  'Enid',
  'Elk City',
  'Seiling',
  'Moore',
  'Lebanon',
  'Blackwell'],
 'OR': ['Canyonville',
  'Portland',
  'Lincoln City',
  'Bend',
  'Woodburn',
  'Newport',
  'Eugene',
  'Gresham',
  'Forest Grove'],
 'PA': ['Harleigh',
  'Warren',
  'Erie',
  'Harrisburg',
  'Altoona',
  'Ramey',
  'Palermo',
  'Mercer',
  'Tannersville',
  'Cranberry Twp',
  'East Hickory',
  'Mendenhall',
  'Huntingdon',
  'Indiana'],
 'RI': ['Providence'],
 'SC': ['Columbia',
  'Dillon',
  'Aiken',
  'Georgetown',
  'Abbeville',
  'Irmo',
  'Blythewood',
  'Mullins',
  'Boiling Springs',
  'Little River',
  'Florence'],
 'SD': ['Brookings', 'Spearfish'],
 'TN': ['Lenoir City',
  'Soddy Daisy',
  'Madisonville',
  'Nashville',
  'Johnson City',
  'Kingston',
  'Chattanooga',
  'Jackson',
  'Knoxville',
  'Clinton',
  'Sweetwater',
  'Cleveland',
  'Hendersonville',
  'Crossville',
  'Gatlinburg'],
 'TX': ['Houston',
  'Beaumont',
  'Mont Belvieu',
  'Grand Prairie',
  'Tyler',
  'Waco',
  'Austin',
  'San Antonio',
  'Dallas',
  'Abilene',
  'Fort Worth',
  'Irving',
  'Carrollton',
  'El Paso',
  'San Marcos',
  'Corpus Christi',
  'Carrizo Springs',
  'Lubbock',
  'Celina',
  'Port Aransas',
  'Junction'],
 'UT': ['Eden', 'Park City', 'Springdale', 'Midway'],
 'VT': [],
 'VA': ['Virginia Beach',
  'Leesburg',
  'Alexandria',
  'Manassas',
  'Roanoke',
  'Ruther Glen',
  'Ashburn',
  'Springfield',
  'Dublin',
  'Richmond',
  'Arlington',
  'Tappahannock',
  'Max Meadows',
  'Chantilly',
  'Emporia'],
 'WA': ['Spokane',
  'Winthrop',
  'Auburn',
  'Seattle',
  'Medina',
  'Friday Harbor',
  'Woodland',
  'Oak Harbor'],
 'WV': ['Charles Town'],
 'WI': ['Fremont',
  'Random Lake',
  'Appleton',
  'Williams Bay',
  'Eau Claire',
  'Genoa City',
  'Oconomowoc',
  'Fond Du Lac',
  'Kenosha',
  'Black River Falls',
  'Green Bay'],
 'WY': ['Cheyenne', 'Cody', 'Pinedale']}