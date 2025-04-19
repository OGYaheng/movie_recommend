import pandas as pd
from sklearn.neighbors import NearestNeighbors

# 載入電影資料集（用戶評分）
movie_data = pd.read_csv('movies.csv')  # 包含欄位: movie_id, title
ratings_data = pd.read_csv('ratings.csv')  # 包含欄位: user_id, movie_id, rating

# 創建用戶-電影評分矩陣（pivot 表格）
user_movie_ratings = ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# 使用 Nearest Neighbors 算法來做推薦
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_movie_ratings.values.T)

def get_movie_id_by_title(title):
    """根據電影名稱獲得電影ID"""
    movie_id = movie_data[movie_data['title'].str.contains(title, case=False, na=False)]
    if not movie_id.empty:
        return movie_id.iloc[0]['movie_id']
    return None

def recommend_movies(movie_title, n_recommendations=5):
    """根據電影名稱給出推薦的電影"""
    movie_id = get_movie_id_by_title(movie_title)
    
    if not movie_id:
        return f"找不到名為 '{movie_title}' 的電影，請檢查電影名稱是否正確。"
    
    movie_index = user_movie_ratings.columns.get_loc(movie_id)
    
    distances, indices = model.kneighbors(user_movie_ratings.values.T[movie_index].reshape(1, -1), n_neighbors=n_recommendations + 1)
    
    recommended_movies = []
    for i in range(1, len(indices.flatten())):
        recommended_movie = movie_data[movie_data['movie_id'] == user_movie_ratings.columns[indices.flatten()[i]]]['title'].values[0]
        recommended_movies.append(recommended_movie)
    
    return recommended_movies

# 主程式：要求用戶輸入電影名稱並給出推薦
def main():
    print("歡迎來到電影推薦系統！")
    movie_title = input("請輸入你喜歡的電影名稱: ")
    
    recommended_movies = recommend_movies(movie_title)
    
    if isinstance(recommended_movies, list):
        print("\n以下是基於你的電影偏好，推薦給你的電影：")
        for i, movie in enumerate(recommended_movies, 1):
            print(f"{i}. {movie}")
    else:
        print(recommended_movies)

if __name__ == "__main__":
    main()