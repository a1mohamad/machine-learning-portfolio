from fastapi import FastAPI, UploadFile, HTTPException
from fastapi import File, Form, Query
import pandas as pd
import pickle
from app.inference import predict_movie, single_movie_pred
from app.schemas import BatchPredictionResponse, SinglePredictionResponse, SingleMovieInput
from sqlalchemy import create_engine
from sqlalchemy import inspect
from app.utils import replace_nan_with_none, log_prediction

# Database connection settings
DB_USER = "postgres"
DB_PASSWORD = "math3141"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "tmdb"

# Create a SQLAlchemy engine for database access
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# FastAPI application instance
app = FastAPI()

# Load train_dict at app startup
with open("models/train_dict.pkl", "rb") as f:
    train_dict = pickle.load(f)

@app.post("/predict/movies_file", response_model=BatchPredictionResponse)
def predict_post(
    csv_file: UploadFile = File(None),
    table : str = Form(None)
):
    """
    Batch prediction endpoint (POST).
    Accepts either a CSV file upload or a database table name via form data.
    - Only one of csv_file or table must be provided.
    - Returns revenue predictions for all movies in the file/table.
    """
    # Normalize blank values
    if table == "" or table is None:
        table = None
    if csv_file is not None and getattr(csv_file, "filename", "") == "":
        csv_file = None

    # Only one input allowed
    if csv_file is not None and table is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of: csv_file or table."
        )
    if csv_file is None and table is None:
        raise HTTPException(
            status_code=400,
            detail="You must provide either 'csv_file' or 'table'!"
        )
    # Handle CSV upload
    if csv_file:
        try:
            df = pd.read_csv(csv_file.file)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read uploaded CSV file: {str(e)}"
            )
    # Handle fetching from database table
    elif table:
        try:
            df = pd.read_sql_table(table, engine)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read table {table}: {str(e)}"
            )
    # Predict revenue and return results
    preds = predict_movie(df, train_dict)
    preds = [round(float(p), 2) for p in preds]
    df = df.where(pd.notnull(df), None)  # Replace NaN with None for logging
    # Log each prediction to the database
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict = replace_nan_with_none(row_dict)
        try:
            log_prediction(
                engine=engine,
                endpoint="/predict/movies_file",
                input_data=row_dict,
                prediction=preds[i],
                imdb_id=row.get("imdb_id"),
                status="success"
                # you can add source=source if you want to log where the data came from
            )
        except Exception as e:
            print(f'logging error: {e}')
            pass
    return {"revenue_predictions": preds}

@app.get("/predict/movies_file", response_model=BatchPredictionResponse)
def predict_get(
    csv_file: str = Query(None, description="Path to the CSV file to server"),
    table: str = Query(None, description="Database table name"),
):
    """
    Batch prediction endpoint (GET).
    Accepts either a CSV file path or a database table name as query parameters.
    - Only one of csv_file or table must be provided.
    - Returns revenue predictions for all movies in the file/table.
    """
    # Handle CSV file path
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV file {csv_file}: {str(e)}"
            )
    # Handle fetching from database table
    elif table:
        try:
            df = pd.read_sql_table(table, engine)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read table {table}: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="You must provide either 'csv_path' or 'table' as a query parameter!"
        )
    # Predict revenue and return results
    preds = predict_movie(df, train_dict)
    preds=[round(float(p), 2) for p in preds]
    df = df.where(pd.notnull(df), None)  # Replace NaN with None for logging
    # Log each prediction to the database
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict = replace_nan_with_none(row_dict)
        try:
            log_prediction(
                engine=engine,
                endpoint="/predict/movies_file",
                input_data=row_dict,
                prediction=preds[i],
                imdb_id=row.get("imdb_id"),
                status="success"
                # you can add source=source if you want to log where the data came from
            )
        except Exception as e:
            print(f'logging error: {e}')
            pass

    return {"revenue_predictions": preds}

@app.post("/predict/single_movie", response_model= SinglePredictionResponse)
def predict_single(movie: SingleMovieInput):
    """
    Single movie prediction endpoint (POST).
    Accepts a single movie's data as JSON.
    Returns the revenue prediction for this movie.
    """
    try:
        movie_dict = movie.dict()
        imdb_id = movie_dict.get("imdb_id")
        pred = single_movie_pred(movie_dict)
        pred = round(pred, 2)
        log_prediction(
            engine=engine,
            endpoint="/predict/single_movie",
            input_data=movie_dict,
            prediction=pred,
            imdb_id=imdb_id,
            status="success"
        )
        return {"revenue_prediction": pred}
    except Exception as e:
        log_prediction(
            engine=engine,
            endpoint="/predict/single_movie",
            input_data=movie_dict,
            prediction=None,
            imdb_id=imdb_id,
            status="failure",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to predict revenue for the provided movie data: {str(e)}"
        )

@app.get("/predict/movie_from_db_by_imdbID", response_model=SinglePredictionResponse)
def predict_movie_by_imdbID(
    imdb_id: str = Query(..., description="IMDB ID of the movie to predict revenue for")    
):
    """
    Single movie prediction by IMDb ID (GET).
    Looks up a movie by IMDb ID in all available database tables, provided the table contains all required fields.
    Returns the revenue prediction for that movie.
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        # List all required fields for the model/pipeline
        REQUIRED_FIELDS = ["imdb_id", "rating", "totalVotes", "release_date"]
        for table in tables:
            columns = [col['name'] for col in inspector.get_columns(table)]
            # Only process tables with all required columns
            if all(field in columns for field in REQUIRED_FIELDS):
                query = f"SELECT * FROM {table} WHERE imdb_id = %s Limit 1"
                df = pd.read_sql_query(query, engine, params=(imdb_id,))
                if not df.empty:
                    movie_dict = df.iloc[0].to_dict()
                    movie_dict = replace_nan_with_none(movie_dict)
                    # Defensive check for all required fields
                    missing = [f for f in REQUIRED_FIELDS if f not in movie_dict]
                    if missing:
                        raise HTTPException(
                            status_code=422,
                            detail=f"Movie found but missing fields: {missing}"
                        )
                    pred = single_movie_pred(movie_dict)
                    pred = round(pred, 2)
                    # LOG THE PREDICTION
                    log_prediction(
                        engine=engine,
                        endpoint="/predict/movie_from_db_by_imdbID",
                        input_data=movie_dict,
                        prediction=pred,
                        imdb_id=imdb_id,
                        status="success"
                    )
                    return {"revenue_prediction": pred}
        # No matching movie found
        raise HTTPException(
            status_code=404,
            detail=f"Movie with IMDB ID {imdb_id} not found in any table."
        )
    except Exception as e:
        # Log the failure
        log_prediction(
            engine=engine,
            endpoint="/predict/movie_from_db_by_imdbID",
            input_data={"imdb_id": imdb_id},
            prediction=None,
            imdb_id=imdb_id,
            status="failure",
            error_message=str(e)
        )
        # Catch any error in the pipeline/model
        raise HTTPException(
            status_code=400,
            detail=f"Failed to predict revenue for movie with IMDB ID {imdb_id}: {str(e)}"
        )  