# A dictionary of movie critics and their ratings of a small
# set of movies
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
'You, Me and Dupree': 3.5},
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
'The Night Listener': 4.5, 'Superman Returns': 4.0,
'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
'You, Me and Dupree': 2.0},
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

from numpy import *
from math import sqrt
from operator import itemgetter

# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
  # Get the list of shared_items
  si={}
  for item in prefs[person1]:
    if item in prefs[person2]: si[item]=1

  # if they have no ratings in common, return 0
  if len(si)==0: return 0

  # Add up the squares of all the differences
  sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
                      for item in prefs[person1] if item in prefs[person2]])

  return 1/(1+sum_of_squares)

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
  # Get the list of mutually rated items
  si={}
  for item in prefs[p1]:
    if item in prefs[p2]: si[item]=1

  # if they are no ratings in common, return 0
  if len(si)==0: return 0

  # Sum calculations
  n=len(si)
  
  # Sums of all the preferences
  sum1=sum([prefs[p1][it] for it in si])
  sum2=sum([prefs[p2][it] for it in si])
  
  # Sums of the squares
  sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
  sum2Sq=sum([pow(prefs[p2][it],2) for it in si])	
  
  # Sum of the products
  pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
  
  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  if den==0: return 0

  r=num/den

  return r


#Computes the Manhattan distance.
def manhattan(prefs,p1,p2): 
    distance = 0 
    commonRatings = False
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: 
            si[item]=1
            distance+=abs(prefs[p1][item]-prefs[p2][item])
            commonRatings = True
        if commonRatings:
            return distance
        else:
            return -1


# Returns the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
  scores=[(similarity(prefs,person,other),other)
                  for other in prefs if other!=person]
  scores.sort()
  scores.reverse()
  return scores[0:n]

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
  totals={}
  simSums={}
  for other in prefs:
    # don't compare me to myself
    if other==person: continue
    sim=similarity(prefs,person,other)

    # ignore scores of zero or lower
    if sim<=0: continue
    for item in prefs[other]:
      # only score movies I haven't seen yet
      if item not in prefs[person] or prefs[person][item]==0:
        # Similarity * Score
        totals.setdefault(item,0)
        totals[item]+=prefs[other][item]*sim
        # Sum of similarities
        simSums.setdefault(item,0)
        simSums[item]+=sim

  # Create the normalized list
  rankings=[(total/simSums[item],item) for item,total in totals.items()]

  # Return the sorted list
  rankings.sort()
  rankings.reverse()
  return rankings

def transformPrefs(prefs):
  result={}
  for person in prefs:
    for item in prefs[person]:
      result.setdefault(item,{})
      
      # Flip item and person
      result[item][person]=prefs[person][item]
  return result


def calculateSimilarItems(prefs,n=10):
  # Create a dictionary of items showing which other items they
  # are most similar to.
  result={}
  # Invert the preference matrix to be item-centric
  itemPrefs=transformPrefs(prefs)
  c=0
  for item in itemPrefs:
    # Status updates for large datasets
    c+=1
    if c%100==0: print "%d / %d" % (c,len(itemPrefs))
    # Find the most similar items to this one
    scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
    result[item]=scores
  return result

def getRecommendedItems(prefs,itemMatch,user):
  userRatings=prefs[user]
  scores={}
  totalSim={}
  # Loop over items rated by this user
  for (item,rating) in userRatings.items( ):
    # Loop over items similar to this one
    for (similarity,item2) in itemMatch[item]:
      # Ignore if this user has already rated this item
      if item2 in userRatings: continue
      # Weighted sum of rating times similarity
      scores.setdefault(item2,0)
      scores[item2]+=similarity*rating
      # Sum of all the similarities
      totalSim.setdefault(item2,0)
      totalSim[item2]+=similarity

  # Divide each total score by total weighting to get an average
  rankings=[(score/totalSim[item],item) for item,score in scores.items( )]

  # Return the rankings from highest to lowest
  rankings.sort( )
  rankings.reverse( )
  return rankings

def loadMovieLens(path='\home\dayle\Desktop\python\data\movielens'):
  # Get movie titles
  movies={}
  for line in open(path+'/u.item'):
    (id,title)=line.split('|')[0:2]
    movies[id]=title
  
  # Load data
  prefs={}
  for line in open(path+'/u.data'):
    (user,movieid,rating,ts)=line.split('\t')
    prefs.setdefault(user,{})
    prefs[user][movies[movieid]]=float(rating)
  return prefs


def getdistances(data,vec1):
    distancelist=[]
    for i in range(len(data)):
        vec2=data[i]['input']
        distancelist.append((euclidean(vec1,vec2),i))
    distancelist.sort( )
    return distancelist


def gaussian(dist,sigma=10.0):
    return math.e**(-dist**2/(2*sigma**2))

def weightedknn(data,vec1,k=5,weightf=gaussian):
    # Get distances
    dlist=getdistances(data,vec1)
    avg=0.0
    totalweight=0.0
    # Get weighted average
    for i in range(k):
        dist=dlist[i][0]
        idx=dlist[i][1]
        weight=weightf(dist)
        avg+=weight*data[idx]['result']
        totalweight+=weight
    avg=avg/totalweight
    return avg

#tf-idf section
def freq(prefs, person):
    return prefs.split(None).count(person)
    
def wordcount(prefs):
    return len(prefs.split(None))
    
def numDocsContaining(prefs, person):
    count=0
    for preference in prefs:
        if freq(preference, person)>0:
            count+=1
    return count

def tf(prefs,  person):
    return (freq(prefs, person) / float(wordcount(prefs)))
    
def idf(prefs, person):
    return math.log(len(prefs) / float(numDocsContaining(prefs, person)))
    
    
def tf_idf(prefs,person):
    return (tf(prefs, person) * idf(prefs, person))


#creates a sorted list of users based on their distance to username
def compute_nearest_neighbor(username, users, k=3):
    distances = [] 
    for user in users: 
        if user != username: 
            distance = manhattan(users, username,  user) 
            distances.append((distance, user)) 
            # sort based on distance -- closest first
            distances.sort() 
    return distances[0:k]


def average_movie_rating(movie_id):
    movie_dict = {}
    for line in open('/home/dayle/Desktop/python/data/rec_sys/movielens/u.item'):
        split_line = line.split('|')
        movie_dict[split_line[0]] = {
            'title': split_line[1],
            'release date': split_line[2],
            'video release date': split_line[3],
            'IMDB URL': split_line[4],
        }

    user_ratings = {}
    for uline in open('/home/dayle/Desktop/python/data/rec_sys/movielens/u.data'):
        split_uline = uline.split()
        movie_name = movie_dict[split_uline[1]]['title']
        if user_ratings.get(movie_name):
            user_ratings[movie_name][split_uline[0]] = float(split_uline[2])
        else:	
            user_ratings[movie_name] = {
                split_uline[0]: float(split_uline[2])
             }
    
    movie_name = movie_dict[str(movie_id)]['title']
    if user_ratings.get(movie_name):
        # average = sum getvalues / count getvalues??
        total = 0
        count = 0
        for key in user_ratings[movie_name]:
            total += user_ratings[movie_name][key]
            count += 1
        print ("Average movie rating for %s is %.2f" %(movie_name,  round(total/count,2)))
    else:
        print "No user ratings for this movie"

	
def knn(movie_ratings, predict_movie, prev_rated,num_neighbors=3):
    #prev_rated is a dictionary of previously rated movies (movie_id: rating)
    eucl_dist = {}
    for k,v in prev_rated:
        print prev_rated
    eucl_dist[str(dist(movie_ratings,predict_movie,v))] = (k, v)
   
    # sort keys of eucl_dict
    eucl_keys = [int(x) for x in eucl_dist.keys()]
    sorted_keys = sorted(eucl_keys)

    neighbors = sorted_keys[0:num_neighbors]
    user_rating_total = 0
    for neighbor in neighbors:
        user_rating_total += eucl_dist[neighbor][1]
    knn_est = user_rating_total/num_neighbors

    return knn_est


# Returns the average absolute error between n pairs
def mean_abs_error(prefs,person1,person2):
  # Get the list of shared_items
  si={}
  for item in prefs[person1]:
    if item in prefs[person2]: si[item]=1

  # if they have no ratings in common, return 0
  if len(si)==0: return 0

  # Sum calculations
  n=len(si)

  # Add up the squares of all the differences
  sum_of_items=sum([abs(prefs[person1][item]-prefs[person2][item])
                      for item in prefs[person1] if item in prefs[person2]])

  return sum_of_items/n


#Compute Root Mean Squared Error. 
def compute_rmse(prefs,person1,person2):
    # Get the list of shared_items
  si={}
  for item in prefs[person1]:
    if item in prefs[person2]: si[item]=1

  # if they have no ratings in common, return 0
  if len(si)==0: return 0

  # Sum calculations
  n=len(si)

  # Add up the squares of all the differences
  sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item], 2)
                      for item in prefs[person1] if item in prefs[person2]])
  #return the computed RMSE
  return np.sqrt(sum_of_squares/n)
